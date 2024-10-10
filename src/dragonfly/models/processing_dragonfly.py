import ast
import math
import random
import re
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy
from transformers.utils import (
    TensorType,
    is_torch_available,
    logging,
    requires_backends,
)

if is_torch_available():
    # from .image_processing_fuyu import FuyuBatchFeature
    from transformers.models.fuyu.image_processing_fuyu import FuyuBatchFeature


logger = logging.get_logger(__name__)


if is_torch_available():
    import torch


TEXT_REPR_BBOX_OPEN = "<box>"
TEXT_REPR_BBOX_CLOSE = "</box>"
TEXT_REPR_POINT_OPEN = "<point>"
TEXT_REPR_POINT_CLOSE = "</point>"

IMAGE_PLACEHOLDER_TOKEN = "<|reserved_special_token_0|>"
IMAGE_NEWLINE_TOKEN = "<|reserved_special_token_1|>"

TOKEN_BBOX_OPEN_STRING = "<|reserved_special_token_2|>"  # <bbox>
TOKEN_BBOX_CLOSE_STRING = "<|reserved_special_token_3|>"  # </bbox>
TOKEN_POINT_OPEN_STRING = "<|reserved_special_token_4|>"  # <point>
TOKEN_POINT_CLOSE_STRING = "<|reserved_special_token_5|>"  # </point>
BEGINNING_OF_ANSWER_STRING = "<|end_header_id|>"  # <boa>

LOW_IMG_EMB_LENGTH = 1
LOW_IMG_PATCH_SIZE = 577
HIGH_IMG_EMB_LENGTH = 40
HIGH_IMG_PATCH_SIZE = 37

def full_unpacked_stream_to_tensor(
    all_bi_tokens_to_place: List[int],
    full_unpacked_stream: List["torch.Tensor"],
    fill_value: int,
    batch_size: int,
    new_seq_len: int,
    offset: int,
) -> "torch.Tensor":
    """Takes an unpacked stream of tokens (i.e. a list of tensors, one for each item in the batch) and does
    the required padding to create a single tensor for the batch of shape batch_size x new_seq_len.
    """

    assert len(all_bi_tokens_to_place) == batch_size
    assert len(full_unpacked_stream) == batch_size

    # Create padded tensors for the full batch.
    new_padded_tensor = torch.full(
        [batch_size, new_seq_len],
        fill_value=fill_value,
        dtype=full_unpacked_stream[0].dtype,
        device=full_unpacked_stream[0].device,
    )

    # Place each batch entry into the batch tensor.
    for bi in range(batch_size):
        tokens_to_place = all_bi_tokens_to_place[bi]
        new_padded_tensor[bi, :tokens_to_place] = full_unpacked_stream[bi][offset : tokens_to_place + offset]

    return new_padded_tensor


def construct_full_unpacked_stream(
    num_real_text_tokens: Union[List[List[int]], "torch.Tensor"],
    input_stream: "torch.Tensor",
    image_tokens: List[List["torch.Tensor"]],
    batch_size: int,
    num_sub_sequences: int,
) -> List["torch.Tensor"]:
    """Takes an input_stream tensor of shape B x S x ?. For each subsequence, adds any required
    padding to account for images and then unpacks the subsequences to create a single sequence per item in the batch.
    Returns a list of tensors, one for each item in the batch."""

    all_bi_stream = []

    for batch_index in range(batch_size):
        all_si_stream = []

        # First, construct full token stream (including image placeholder tokens) and loss mask for each subsequence
        # and append to lists. We use lists rather than tensors because each subsequence is variable-sized.
        # TODO Remove this logic in a subsequent release since subsequences are not supported.
        image_adjustment = image_tokens[batch_index][0]
        subsequence_stream = torch.cat([image_adjustment, input_stream[batch_index, 0]], dim=0)
        num_real_tokens = image_adjustment.shape[0] + num_real_text_tokens[batch_index][0]
        all_si_stream.append(subsequence_stream[:num_real_tokens])
        all_bi_stream.append(torch.cat(all_si_stream, dim=0))

    return all_bi_stream


# Vision encoding processing
def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


def resize_and_pad_image(image, target_resolution):
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    new_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image


def divide_to_patches(image, patch_size):
    """
    Divides an image into patches of a specified size.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    """
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)

    return patches


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (tuple): The size of the input image in the format (width, height).
        grid_pinpoints (str): A string representation of a list of possible resolutions.
        patch_size (int): The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    width, height = select_best_resolution(image_size, possible_resolutions)
    return width // patch_size, height // patch_size


def process_anyres_image(image, processor, shortest_edge=None, possible_resolutions=[(4, 3), (3, 4)]):
    """
    Process an image with variable resolutions.

    Args:
        image (PIL.Image.Image): The input image to be processed.
        processor: The image processor object.
        grid_pinpoints (str): A string representation of a list of possible resolutions.

    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    if shortest_edge is None:
        se = processor.size["shortest_edge"]
    else:
        se = shortest_edge
    middle_resolution = select_best_resolution(image.size, [(2 * se, 2 * se), (1 * se, 4 * se), (4 * se, 1 * se)])
    middle_image_padded = image.resize(middle_resolution)
    middle_patches = divide_to_patches(middle_image_padded, se)

    high_resolution = (middle_resolution[0]*3, middle_resolution[1]*3)
    high_image_padded = image.resize(high_resolution)
    high_patches = divide_to_patches(high_image_padded, se)

    image_original_resize = image.resize((se, se))
    image_patches = [image_original_resize] + middle_patches + high_patches
    image_patches = [processor.preprocess(image_patch, return_tensors='pt')['pixel_values'][0]
                     for image_patch in image_patches]
    return torch.stack(image_patches, dim=0), [torch.tensor(1.0)]


def _replace_string_repr_with_token_tags(prompt: str) -> str:
    prompt = prompt.replace(TEXT_REPR_POINT_OPEN, TOKEN_POINT_OPEN_STRING)
    prompt = prompt.replace(TEXT_REPR_POINT_CLOSE, TOKEN_POINT_CLOSE_STRING)
    prompt = prompt.replace(TEXT_REPR_BBOX_OPEN, TOKEN_BBOX_OPEN_STRING)
    prompt = prompt.replace(TEXT_REPR_BBOX_CLOSE, TOKEN_BBOX_CLOSE_STRING)
    return prompt


def _segment_prompt_into_text_token_conversions(prompt: str) -> List:
    """
    Given a string prompt, converts the prompt into a list of TextTokenConversions.
    """
    # Wherever, we notice the [TOKEN_OPEN_STRING, TOKEN_CLOSE_STRING], we split the prompt
    prompt_text_list: List = []
    regex_pattern = re.compile(f"({TOKEN_BBOX_OPEN_STRING}|{TOKEN_BBOX_CLOSE_STRING}|{TOKEN_POINT_OPEN_STRING}|{TOKEN_POINT_CLOSE_STRING})")
    # Split by the regex pattern
    prompt_split = regex_pattern.split(prompt)
    for i, elem in enumerate(prompt_split):
        if len(elem) == 0 or elem in [
            TOKEN_BBOX_OPEN_STRING,
            TOKEN_BBOX_CLOSE_STRING,
            TOKEN_POINT_OPEN_STRING,
            TOKEN_POINT_CLOSE_STRING,
        ]:
            continue
        prompt_text_list.append((elem, i > 1 and prompt_split[i - 1] in [TOKEN_BBOX_OPEN_STRING, TOKEN_POINT_OPEN_STRING]))
    return prompt_text_list


def _transform_coordinates_and_tokenize(prompt: str, scale_factor: float, tokenizer) -> List[int]:
    """
    This function transforms the prompt in the following fashion:
    - <box> <point> and </box> </point> to their respective token mappings
    - extract the coordinates from the tag
    - transform the coordinates into the transformed image space
    - return the prompt tokens with the transformed coordinates and new tags

    Bounding boxes and points MUST be in the following format: <box>y1, x1, y2, x2</box> <point>x, y</point> The spaces
    and punctuation added above are NOT optional.
    """
    # Make a namedtuple that stores "text" and "is_bbox"

    # We want to do the following: Tokenize the code normally -> when we see a point or box, tokenize using the tokenize_within_tag function
    # When point or box close tag, continue tokenizing normally
    # First, we replace the point and box tags with their respective tokens
    prompt = _replace_string_repr_with_token_tags(prompt)
    # Tokenize the prompt
    # Convert prompt into a list split
    prompt_text_list = _segment_prompt_into_text_token_conversions(prompt)
    transformed_prompt_tokens: List[int] = []
    for elem in prompt_text_list:
        if elem[1]:
            # This is a location, we need to tokenize it
            within_tag_tokenized = _transform_within_tags(elem[0], scale_factor, tokenizer)
            # Surround the text with the open and close tags
            transformed_prompt_tokens.extend(within_tag_tokenized)
        else:
            transformed_prompt_tokens.extend(tokenizer(elem[0], add_special_tokens=True).input_ids)
    return transformed_prompt_tokens


def _transform_within_tags(text: str, scale_factor: float, tokenizer) -> List[int]:
    """
    Given a bounding box of the fashion <box>1, 2, 3, 4</box> | <point>1, 2</point> This function is responsible for
    converting 1, 2, 3, 4 into tokens of 1 2 3 4 without any commas.
    """
    # Convert the text into a list of strings.
    num_int_strs = text.split(",")
    if len(num_int_strs) == 2:
        # If there are any open or close tags, remove them.
        token_space_open_string = tokenizer.vocab[TOKEN_POINT_OPEN_STRING]
        token_space_close_string = tokenizer.vocab[TOKEN_POINT_CLOSE_STRING]
    else:
        token_space_open_string = tokenizer.vocab[TOKEN_BBOX_OPEN_STRING]
        token_space_close_string = tokenizer.vocab[TOKEN_BBOX_CLOSE_STRING]

    # Remove all spaces from num_ints
    num_ints = [float(num.strip()) for num in num_int_strs]
    # scale to transformed image siz
    if len(num_ints) == 2:
        num_ints_translated = scale_point_to_transformed_image(x=num_ints[0], y=num_ints[1], scale_factor=scale_factor)
    elif len(num_ints) == 4:
        num_ints_translated = scale_bbox_to_transformed_image(
            top=num_ints[0],
            left=num_ints[1],
            bottom=num_ints[2],
            right=num_ints[3],
            scale_factor=scale_factor,
        )
    else:
        raise ValueError(f"Invalid number of ints: {len(num_ints)}")
    # Tokenize the text, skipping the
    tokens = [tokenizer.vocab[str(num)] for num in num_ints_translated]
    return [token_space_open_string] + tokens + [token_space_close_string]


def _tokenize_prompts_with_image_and_batch(
    tokenizer,
    prompts: List[List[str]],
    scale_factors: Optional[List[List["torch.Tensor"]]],
    max_tokens_to_generate: int,
    max_position_embeddings: int,
    add_BOS: bool,  # Same issue with types as above
    add_beginning_of_answer_token: bool,
    max_length: int = 1024,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """
    Given a set of prompts and number of tokens to generate:
    - tokenize prompts
    - set the sequence length to be the max of length of prompts plus the number of tokens we would like to generate
    - pad all the sequences to this length so we can convert them into a 3D tensor.
    """

    text_max_length = int(max_length / 2)
    # print(text_max_length)
    # If not tool use, tranform the coordinates while tokenizing
    transformed_prompt_tokens = [[tokenizer(prompt, add_special_tokens=True).input_ids for prompt in prompt_seq] for prompt_seq in prompts]

    transformed_prompt_tokens = [[prompt[: text_max_length - 1] for prompt in prompt_seq] for prompt_seq in transformed_prompt_tokens]

    prompts_tokens = transformed_prompt_tokens

    if add_BOS:
        bos_token = tokenizer.vocab["<|begin_of_text|>"]
    else:
        bos_token = tokenizer.vocab["<|end_of_text|>"]
    # prompts_tokens = [[[bos_token] + x for x in prompt_seq] for prompt_seq in prompts_tokens]
    if add_beginning_of_answer_token:
        boa = tokenizer.vocab[BEGINNING_OF_ANSWER_STRING]
        # Only add bbox open token to the last subsequence since that is what will be completed
        for token_seq in prompts_tokens:
            token_seq[-1].append(boa)
    else:
        boa = tokenizer.vocab["<|end_of_text|>"]
        # Only add bbox open token to the last subsequence since that is what will be completed
        for token_seq in prompts_tokens:
            token_seq[-1].append(boa)

    # Now we have a list of list of tokens which each list has a different
    # size. We want to extend this list to:
    #   - incorporate the tokens that need to be generated
    #   - make all the sequences equal length.
    # Get the prompts length.

    prompts_length = [[len(x) for x in prompts_tokens_seq] for prompts_tokens_seq in prompts_tokens]
    # Get the max prompts length.
    max_prompt_len: int = np.max(prompts_length)
    # Number of tokens in the each sample of the batch.
    samples_length = min(max_prompt_len + max_tokens_to_generate, max_position_embeddings)
    if max_prompt_len + max_tokens_to_generate > max_position_embeddings:
        logger.warning(
            f"Max subsequence prompt length of {max_prompt_len} + max tokens to generate {max_tokens_to_generate}",
            f"exceeds context length of {max_position_embeddings}. Will generate as many tokens as possible.",
        )
    # Now update the list of list to be of the same size: samples_length.
    for prompt_tokens_seq, prompts_length_seq in zip(prompts_tokens, prompts_length):
        for prompt_tokens, prompt_length in zip(prompt_tokens_seq, prompts_length_seq):
            if len(prompt_tokens) > samples_length:
                raise ValueError("Length of subsequence prompt exceeds sequence length.")
            padding_size = samples_length - prompt_length
            prompt_tokens.extend([tokenizer.vocab["<|end_of_text|>"]] * padding_size)

    # Now we are in a structured format, we can convert to tensors.
    prompts_tokens_tensor = torch.tensor(prompts_tokens, dtype=torch.int64)
    prompts_length_tensor = torch.tensor(prompts_length, dtype=torch.int64)

    return prompts_tokens_tensor, prompts_length_tensor


# Simplified assuming self.crop_top = self.padding_top = 0
def original_to_transformed_h_coords(original_coords, scale_h):
    return np.round(original_coords * scale_h).astype(np.int32)


# Simplified assuming self.crop_left = self.padding_left = 0
def original_to_transformed_w_coords(original_coords, scale_w):
    return np.round(original_coords * scale_w).astype(np.int32)


def scale_point_to_transformed_image(x: float, y: float, scale_factor: float) -> List[int]:
    x_scaled = original_to_transformed_w_coords(np.array([x / 2]), scale_factor)[0]
    y_scaled = original_to_transformed_h_coords(np.array([y / 2]), scale_factor)[0]
    return [x_scaled, y_scaled]


def scale_bbox_to_transformed_image(top: float, left: float, bottom: float, right: float, scale_factor: float) -> List[int]:
    top_scaled = original_to_transformed_w_coords(np.array([top / 2]), scale_factor)[0]
    left_scaled = original_to_transformed_h_coords(np.array([left / 2]), scale_factor)[0]
    bottom_scaled = original_to_transformed_w_coords(np.array([bottom / 2]), scale_factor)[0]
    right_scaled = original_to_transformed_h_coords(np.array([right / 2]), scale_factor)[0]
    return [top_scaled, left_scaled, bottom_scaled, right_scaled]


class DragonflyProcessor(ProcessorMixin):
    r"""
    Constructs a Dragonfly processor which wraps a image processor and a Llama tokenizer into a single processor.

    [`DragonflyProcessor`] offers all the functionalities of [`AutoImageProcessor`] and [`LlamaTokenizerFast`]. See the
    [`~DragonflyProcessor.__call__`] and [`~DragonflyProcessor.decode`] for more information.

    Args:
        image_processor ([`AutoImageProcessor`]):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`]):
            The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor, tokenizer, image_encoding_style="llava-hd"):
        super().__init__(image_processor=image_processor, tokenizer=tokenizer)
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_tokens_to_generate = 10
        self.max_position_embeddings = 16384
        self.pad_token_id = tokenizer.eos_token_id
        self.dummy_image_index = -1
        self.image_encoding_style = image_encoding_style

    def find_and_remove_tokens(self, input_ids, labels, token_id):
        batch_size, seq_len = input_ids.size()

        # Create lists to store the new tensors
        new_input_list = []
        new_labels_list = []

        for i in range(batch_size):
            single_input = input_ids[i, :]
            single_label = labels[i, :]

            # Remove the last token_id
            token_indices = (single_input == token_id).nonzero(as_tuple=True)[0]
            if len(token_indices) > 1:
                last_token_index = token_indices[-1]
                single_input[last_token_index] = self.tokenizer.eos_token_id
                single_label[last_token_index] = self.tokenizer.eos_token_id

            # Append the new sequence to the list
            new_input_list.append(single_input)
            new_labels_list.append(single_label)

        return torch.stack(new_input_list), torch.stack(new_labels_list)

    def get_labels(self, input_ids, special_token_id, masking_number=-100):
        # Initialize labels tensor filled with masking_number
        labels = torch.full_like(input_ids, masking_number)

        # Iterate through each sequence in the batch
        for i in range(input_ids.shape[0]):
            seq = input_ids[i]

            start = (seq == special_token_id).nonzero(as_tuple=False)[0].squeeze()
            # Unmask the tokens between the first and second occurren
            labels[i, start + 1 :] = seq[start + 1 :]

        return labels

    def _right_pad_inputs_with_attention_mask(self, model_inputs: List[Dict], return_attention_mask: bool):
        max_length_input_ids = max(entry["input_ids"].shape[1] for entry in model_inputs)
        max_length_image_patch_indices = max(entry["image_patches_indices"].shape[1] for entry in model_inputs)

        batched_inputs = {"input_ids": [], "image_patches": [], "image_patches_indices": [], "attention_mask": []}

        for entry in model_inputs:
            for key, tensor in entry.items():
                if key == "input_ids":
                    num_padding_tokens = max_length_input_ids - tensor.shape[1]
                    padded_input_ids = torch.cat(
                        [tensor, torch.full((tensor.shape[0], num_padding_tokens), self.pad_token_id, dtype=torch.long)],
                        dim=1,
                    )
                    batched_inputs[key].append(padded_input_ids)

                    attention_mask = torch.cat(
                        [torch.ones_like(tensor), torch.zeros(tensor.shape[0], num_padding_tokens, dtype=torch.long)],
                        dim=1,
                    )
                    batched_inputs["attention_mask"].append(attention_mask)
                elif key == "image_patches":
                    # For image_patches, we don't pad but just append them to the list.
                    batched_inputs[key].append(tensor)
                else:  # for image_patches_indices
                    num_padding_indices = max_length_image_patch_indices - tensor.shape[1]
                    padded_indices = torch.cat(
                        [tensor, torch.full((tensor.shape[0], num_padding_indices), self.dummy_image_index, dtype=torch.long)],
                        dim=1,
                    )
                    batched_inputs[key].append(padded_indices)

        batched_keys = ["input_ids", "image_patches_indices"]
        if return_attention_mask:
            batched_keys.append("attention_mask")
        for key in batched_keys:
            batched_inputs[key] = torch.cat(batched_inputs[key], dim=0)

        return batched_inputs

    def _left_pad_inputs_with_attention_mask(self, model_inputs: List[Dict], return_attention_mask: bool):
        max_length_input_ids = max(entry["input_ids"].shape[1] for entry in model_inputs)
        max_length_image_patch_indices = max(entry["image_patches_indices"].shape[1] for entry in model_inputs)

        batched_inputs = {"input_ids": [], "image_patches": [], "image_patches_indices": [], "attention_mask": []}

        for entry in model_inputs:
            for key, tensor in entry.items():
                if key == "input_ids":
                    num_padding_tokens = max_length_input_ids - tensor.shape[1]
                    padded_input_ids = torch.cat(
                        [
                            torch.full((tensor.shape[0], num_padding_tokens), self.pad_token_id, dtype=torch.long),
                            tensor,
                        ],
                        dim=1,
                    )
                    batched_inputs[key].append(padded_input_ids)

                    attention_mask = torch.cat(
                        [torch.zeros(tensor.shape[0], num_padding_tokens, dtype=torch.long), torch.ones_like(tensor)],
                        dim=1,
                    )
                    batched_inputs["attention_mask"].append(attention_mask)
                elif key == "image_patches":
                    # For image_patches, we don't pad but just append them to the list.
                    batched_inputs[key].append(tensor)
                else:  # for image_patches_indices
                    num_padding_indices = max_length_image_patch_indices - tensor.shape[1]
                    padded_indices = torch.cat(
                        [
                            torch.full((tensor.shape[0], num_padding_indices), self.dummy_image_index, dtype=torch.long),
                            tensor,
                        ],
                        dim=1,
                    )
                    batched_inputs[key].append(padded_indices)
        batched_keys = ["input_ids", "image_patches_indices"]
        if return_attention_mask:
            batched_keys.append("attention_mask")
        for key in batched_keys:
            batched_inputs[key] = torch.cat(batched_inputs[key], dim=0)

        return batched_inputs

    def get_sample_encoding_with_encoder(
        self,
        pixel_values,
        patch_size,
        prompts,
        scale_factors,
        image_placeholder_id,
        image_newline_id,
        add_beginning_of_answer_token,
        max_length,
    ):
        # FIXME max_tokens_to_generate is embedded into this processor's call.
        prompt_tokens, prompts_length = _tokenize_prompts_with_image_and_batch(
            tokenizer=self.tokenizer,
            prompts=prompts,
            scale_factors=scale_factors,
            max_tokens_to_generate=self.max_tokens_to_generate,
            max_position_embeddings=self.max_position_embeddings,
            add_BOS=True,
            add_beginning_of_answer_token=add_beginning_of_answer_token,
            max_length=max_length,
        )

        low_img_emb_length = LOW_IMG_EMB_LENGTH
        low_img_patch_size = LOW_IMG_PATCH_SIZE
        high_img_emb_length = HIGH_IMG_EMB_LENGTH
        high_img_patch_size = HIGH_IMG_PATCH_SIZE

        low_res_newline_idx = [i for i in range(low_img_emb_length * (1 + low_img_patch_size)) if (i + 1) % (low_img_patch_size + 1) == 0]
        high_res_index = max(low_res_newline_idx) + 1
        high_res_newline_idx = [i + high_res_index for i in range(high_img_emb_length * (1 + high_img_patch_size)) if (i + 1) % (high_img_patch_size + 1) == 0]
        
        low_res_input_id_item = torch.full((low_img_emb_length * (1 + low_img_patch_size),), image_placeholder_id, dtype=torch.int32, device=pixel_values.device)
        high_res_input_id_item = torch.full((high_img_emb_length * (1 + high_img_patch_size),), image_placeholder_id, dtype=torch.int32, device=pixel_values.device)

        newline_idx = low_res_newline_idx + high_res_newline_idx
        input_id_item = torch.concat((low_res_input_id_item, high_res_input_id_item))
        input_id_item[newline_idx] = image_newline_id
        placeholder_mask = input_id_item == image_placeholder_id
        patch_indices_per_batch_item = torch.full_like(input_id_item, -1, dtype=torch.int32, device=pixel_values.device)
        patch_idx = torch.arange(low_img_emb_length * low_img_patch_size + high_img_emb_length * high_img_patch_size, dtype=torch.int32, device=pixel_values.device)
        patch_indices_per_batch_item[placeholder_mask] = patch_idx
        image_input_ids = [[input_id_item for _ in prompt_seq] for prompt_seq in prompt_tokens]
        image_patch_indices_per_batch = [[patch_indices_per_batch_item for _ in prompt_seq] for prompt_seq in prompt_tokens]

        image_padded_unpacked_tokens = construct_full_unpacked_stream(
            num_real_text_tokens=prompts_length,
            input_stream=prompt_tokens,
            image_tokens=image_input_ids,
            batch_size=1,
            num_sub_sequences=self.subsequence_length,
        )
        # Construct inputs for image patch indices.
        unpacked_image_patch_indices_per_batch = construct_full_unpacked_stream(
            num_real_text_tokens=prompts_length,
            input_stream=torch.full_like(prompt_tokens, -1),
            image_tokens=image_patch_indices_per_batch,
            batch_size=1,
            num_sub_sequences=self.subsequence_length,
        )
        max_prompt_length = max(x.shape[-1] for x in image_padded_unpacked_tokens)
        max_seq_len_batch = min(max_prompt_length + self.max_tokens_to_generate, self.max_position_embeddings)
        tokens_to_place = min(max_seq_len_batch, max(0, image_padded_unpacked_tokens[0].shape[0]))

        # Use same packing logic for the image patch indices.
        image_patch_input_indices = full_unpacked_stream_to_tensor(
            all_bi_tokens_to_place=[tokens_to_place],
            full_unpacked_stream=unpacked_image_patch_indices_per_batch,
            fill_value=-1,
            batch_size=1,
            new_seq_len=max_seq_len_batch,
            offset=0,
        )

        batch_encoding = {
            "input_ids": image_padded_unpacked_tokens[0].unsqueeze(0),
            "image_patches": pixel_values,
            "image_patches_indices": image_patch_input_indices,
        }
        return batch_encoding

    def __call__(
        self,
        text=None,
        images=None,
        add_special_tokens: bool = True,
        return_attention_mask: bool = True,
        padding: Union[bool, str, PaddingStrategy] = True,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: int = 2048,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_token_type_ids: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        return_tensors: Optional[Union[str, TensorType]] = None,
        add_beginning_of_answer_token: bool = False,
        patch_size=32,
        is_generate=False,
        **kwargs,
    ) -> "FuyuBatchFeature":
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to LlamaTokenizerFast's [`~LlamaTokenizerFast.__call__`] if `text` is not `None` to
        encode the text. To prepare the image(s), this method forwards the `images` and `kwargs` arguments to
        FuyuImageProcessor's [`~FuyuImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            text (`str`, `List[str]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`PIL.Image.Image`, `List[PIL.Image.Image]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.

        Returns:
            [`FuyuBatchEncoding`]: A [`FuyuBatchEncoding`] with the following fields:

            - **input_ids** -- Tensor of token ids to be fed to a model. Returned when `text` is not `None`.
            - **image_patches** -- List of Tensor of image patches. Returned when `images` is not `None`.
            - **image_patches_indices** -- Tensor of indices where patch embeddings have to be inserted by the model.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model when
              `return_attention_mask=True`.
        """
        requires_backends(self, ["torch"])

        # --- Check input validity ---
        if not return_attention_mask:
            raise ValueError("`return_attention_mask=False` is not supported for this model.")
        if text is None and images is None:
            raise ValueError("You have to specify either text or images. Both cannot be None.")
        if text is not None and images is None:
            # logger.warning("You are processing a text with no associated image. Make sure it is intended.")
            self.current_processor = self.tokenizer
            if type(text) == list:
                text = [t + "<|end_of_text|>" for t in text]
            else:
                text = text + "<|end_of_text|>"
            text_encoding = self.tokenizer(
                text=text,
                add_special_tokens=True,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_token_type_ids=return_token_type_ids,
                return_length=return_length,
                verbose=verbose,
                return_tensors=return_tensors,
                **kwargs,
            )
            return text_encoding

        if text is None and images is not None:
            logger.warning("You are processing an image with no associated text. Make sure it is intended.")
            prompts = [[""]]
        if text is not None and images is not None:
            if isinstance(text, str):
                prompts = [[text]]
            elif isinstance(text, list):
                prompts = [[text_seq] for text_seq in text]

        # --- Preprocess images using self.image_processor ---

        # FIXME - We hard code "pt" here because the rest of the processing assumes torch tensors
        if self.image_encoding_style == "llava-hd":
            img_proc_output = [
                process_anyres_image(image, self.image_processor) for image in images
            ]
            batch_images = [item[0] for item in img_proc_output]
            scale_factors = [item[1] for item in img_proc_output]
            self.subsequence_length = 1  # Each batch contains only one sequence.
            self.batch_size = len(batch_images)

            # image_placeholder_id = self.tokenizer(IMAGE_PLACEHOLDER_TOKEN)["input_ids"][1]
            # image_newline_id = self.tokenizer(IMAGE_NEWLINE_TOKEN)["input_ids"][1]

            image_placeholder_id = self.tokenizer.vocab[IMAGE_PLACEHOLDER_TOKEN]
            image_newline_id = self.tokenizer.vocab[IMAGE_NEWLINE_TOKEN]

            all_encodings = []

            for prompt, scale_factor, tensor_batch_image in zip(prompts, scale_factors, batch_images):
                sample_encoding = self.get_sample_encoding_with_encoder(
                    pixel_values=tensor_batch_image,
                    patch_size=patch_size,
                    prompts=[prompt],
                    scale_factors=[scale_factor],
                    image_placeholder_id=image_placeholder_id,
                    image_newline_id=image_newline_id,
                    add_beginning_of_answer_token=add_beginning_of_answer_token,
                    max_length=max_length,
                )
                all_encodings.append(sample_encoding)
            if not is_generate:
                batch_encoding = self._right_pad_inputs_with_attention_mask(model_inputs=all_encodings, return_attention_mask=return_attention_mask)
            else:
                batch_encoding = self._left_pad_inputs_with_attention_mask(model_inputs=all_encodings, return_attention_mask=return_attention_mask)
                batch_encoding["input_ids"] = batch_encoding["input_ids"][:, :-1]
                batch_encoding["image_patches_indices"] = batch_encoding["image_patches_indices"][:, :-1]
                batch_encoding["attention_mask"] = batch_encoding["attention_mask"][:, :-1]

            return FuyuBatchFeature(data=batch_encoding)
        else:
            raise Exception(f"{self.image_encoding_style} encoding style is not supported")

    def post_process_box_coordinates(self, outputs, target_sizes=None):
        """
        Transforms raw coordinates detected by [`FuyuForCausalLM`] to the original images' coordinate space.
        Coordinates will be returned in "box" format, with the following pattern:
            `<box>top, left, bottom, right</box>`

        Point coordinates are not supported yet.

        Args:
            outputs ([`GenerateOutput`]):
                Raw outputs from `generate`.
            target_sizes (`torch.Tensor`, *optional*):
                Tensor of shape (batch_size, 2) where each entry is the (height, width) of the corresponding image in
                the batch. If set, found coordinates in the output sequence are rescaled to the target sizes. If left
                to None, coordinates will not be rescaled.

        Returns:
            `GenerateOutput`: Same output type returned by `generate`, with output token ids replaced with
                boxed and possible rescaled coordinates.
        """

        def scale_factor_to_fit(original_size, target_size=None):
            height, width = original_size
            if target_size is None:
                max_height = self.image_processor.size["height"]
                max_width = self.image_processor.size["width"]
            else:
                max_height, max_width = target_size
            if width <= max_width and height <= max_height:
                return 1.0
            return min(max_height / height, max_width / width)

        def find_delimiters_pair(tokens, start_token, end_token):
            start_id = self.tokenizer.convert_tokens_to_ids(start_token)
            end_id = self.tokenizer.convert_tokens_to_ids(end_token)

            starting_positions = (tokens == start_id).nonzero(as_tuple=True)[0]
            ending_positions = (tokens == end_id).nonzero(as_tuple=True)[0]

            if torch.any(starting_positions) and torch.any(ending_positions):
                return (starting_positions[0], ending_positions[0])
            return (None, None)

        def tokens_to_boxes(tokens, original_size):
            while (pair := find_delimiters_pair(tokens, TOKEN_BBOX_OPEN_STRING, TOKEN_BBOX_CLOSE_STRING)) != (
                None,
                None,
            ):
                start, end = pair
                if end != start + 5:
                    continue

                # Retrieve transformed coordinates from tokens
                coords = self.tokenizer.convert_ids_to_tokens(tokens[start + 1 : end])

                # Scale back to original image size and multiply by 2
                scale = scale_factor_to_fit(original_size)
                top, left, bottom, right = [2 * int(float(c) / scale) for c in coords]

                # Replace the IDs so they get detokenized right
                replacement = f" {TEXT_REPR_BBOX_OPEN}{top}, {left}, {bottom}, {right}{TEXT_REPR_BBOX_CLOSE}"
                replacement = self.tokenizer.tokenize(replacement)[1:]
                replacement = self.tokenizer.convert_tokens_to_ids(replacement)
                replacement = torch.tensor(replacement).to(tokens)

                tokens = torch.cat([tokens[:start], replacement, tokens[end + 1 :]], 0)
            return tokens

        def tokens_to_points(tokens, original_size):
            while (pair := find_delimiters_pair(tokens, TOKEN_POINT_OPEN_STRING, TOKEN_POINT_CLOSE_STRING)) != (
                None,
                None,
            ):
                start, end = pair
                if end != start + 3:
                    continue

                # Retrieve transformed coordinates from tokens
                coords = self.tokenizer.convert_ids_to_tokens(tokens[start + 1 : end])

                # Scale back to original image size and multiply by 2
                scale = scale_factor_to_fit(original_size)
                x, y = [2 * int(float(c) / scale) for c in coords]

                # Replace the IDs so they get detokenized right
                replacement = f" {TEXT_REPR_POINT_OPEN}{x}, {y}{TEXT_REPR_POINT_CLOSE}"
                replacement = self.tokenizer.tokenize(replacement)[1:]
                replacement = self.tokenizer.convert_tokens_to_ids(replacement)
                replacement = torch.tensor(replacement).to(tokens)

                tokens = torch.cat([tokens[:start], replacement, tokens[end + 1 :]], 0)
            return tokens

        if target_sizes is None:
            target_sizes = ((self.image_processor.size["height"], self.image_processor.size["width"]),) * len(outputs)
        elif target_sizes.shape[1] != 2:
            raise ValueError("Each element of target_sizes must contain the size (h, w) of each image of the batch")

        if len(outputs) != len(target_sizes):
            raise ValueError("Make sure that you pass in as many target sizes as output sequences")

        results = []
        for seq, size in zip(outputs, target_sizes):
            seq = tokens_to_boxes(seq, size)
            seq = tokens_to_points(seq, size)
            results.append(seq)

        return results

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)
