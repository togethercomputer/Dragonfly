"""Testing script"""

import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoTokenizer

from dragonfly.models.modeling_dragonfly import DragonflyForCausalLM
from dragonfly.models.processing_dragonfly import DragonflyProcessor
from dragonfly.models.processing_dragonfly import select_best_resolution, resize_and_pad_image, divide_to_patches
from pipeline.train.train_utils import random_seed


def format_text(text, system_prompt=""):
    instruction = f"{system_prompt} {text}" if system_prompt else text
    prompt = f"<|start_header_id|>user<|end_header_id|>\n\n" f"{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    return prompt


def combine_patches(patches, image_size, patch_size, patch_indices):
    """
    Combines patches into the original image.

    Args:
        patches (list): A list of PIL.Image.Image objects representing the patches.
        image_size (tuple): The size of the original image as (width, height).
        patch_size (int): The size of each patch.

    Returns:
        PIL.Image.Image: The reconstructed image.
    """
    width, height = image_size
    original_image = Image.new('RGB', (width, height))
    
    patch_index = 0
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            if patch_index < len(patches):
                patch = patches[patch_index]
                if patch_index in patch_indices:
                    overlay = Image.new('RGBA', (patch_size, patch_size), (255, 255, 0, 64))
                    patch = Image.alpha_composite(patch.convert('RGBA'), overlay)
                original_image.paste(patch.convert('RGB'), (j, i))
                patch_index += 1

    return original_image

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)

# set your model name and image path
pretrained_model_name_or_path = "togethercomputer/Llama-3-8B-Dragonfly-v1"
image_path = "test_images/skateboard.png"
question = "Summarize the visual content of the image."

# For biomed
# pretrained_model_name_or_path = "togethercomputer/Llama-3-8B-Dragonfly-Med-v1"
# image_path = "test_images/ROCO_04197.jpg"
# question = "Provide a brief description of the given image."

# parameters
device = "cuda:0"
seed = 42
temperature = 0


def main():
    random_seed(seed)

    print(f"Loading pretrained model from {pretrained_model_name_or_path}")

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    image_processor = clip_processor.image_processor
    processor = DragonflyProcessor(image_processor=image_processor, tokenizer=tokenizer, image_encoding_style="llava-hd")

    model = DragonflyForCausalLM.from_pretrained(pretrained_model_name_or_path)
    model = model.to(torch.bfloat16)
    model = model.to(device)

    # load the image
    image = Image.open(image_path)
    image = image.convert("RGB")
    images = [image]

    # prepare inputs for the model
    text_prompt = format_text(question)

    # process the text and image
    inputs = processor(text=[text_prompt], images=images, max_length=2048, return_tensors="pt", is_generate=True)
    inputs = inputs.to(device)

    # generate the response
    with torch.inference_mode():
        model_outputs = model(**inputs)
        generation_output = model.generate(**inputs, max_new_tokens=1024, eos_token_id=tokenizer.encode("<|eot_id|>"), do_sample=temperature > 0, temperature=temperature, use_cache=True)
    
    se = image_processor.size["shortest_edge"]
    possible_resolutions = [(6*se, 4*se), (4*se, 6*se), (3*se, 8*se), (8*se, 3*se), (2*se, 12*se), (12*se, 2*se)]
    high_resolution = select_best_resolution(image.size, possible_resolutions)
    high_image_padded = image.resize(high_resolution)
    high_patches = divide_to_patches(high_image_padded, se)

    patch_indices = model_outputs["query_ranks"][0].cpu().tolist()

    highlighted_image = combine_patches(high_patches, high_image_padded.size, se, patch_indices)
    highlighted_image.resize(image.size)
    
    prefix = "/".join(image_path.split(".")[:-1])
    image_type = image_path.split(".")[-1]
    save_path = f"{prefix}_highlighted.{image_type}"
    highlighted_image.save(save_path)

    generation_text = processor.batch_decode(generation_output, skip_special_tokens=False)
    print(generation_text[0].replace("<|reserved_special_token_0|>", "").replace("<|reserved_special_token_1|>", ""))



if __name__ == "__main__":
    main()
