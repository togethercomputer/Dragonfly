import torch

from transformers import FuyuPreTrainedModel, AutoModelForCausalLM, FuyuConfig, PretrainedConfig
from transformers import AutoModelForCausalLM
from transformers import CLIPVisionModel
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers import logging
from transformers import CONFIG_MAPPING

from typing import List, Optional, Tuple, Union
import torch.nn as nn
from torch.nn import functional as F

logger = logging.get_logger(__name__)

class DragonflyConfig(PretrainedConfig):

    model_type = "fuyu"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=128256,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=8192,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=128000,
        eos_token_id=128001,
        tie_word_embeddings=False,
        rope_theta=500000.0,
        attention_dropout=0.0,
        attention_bias=False,
        text_config=None,
        image_size=300,
        patch_size=30,
        num_channels=3,
        rope_scaling=None,
        text_pretrained_model_name_or_path=None,
        image_encoder=None,
        **kwargs,
    ):
        if text_config is None:
            text_config = {
                "vocab_size": vocab_size,
                "hidden_size": hidden_size,
                "intermediate_size": intermediate_size,
                "num_hidden_layers": num_hidden_layers,
                "num_attention_heads": num_attention_heads,
                "num_key_value_heads": num_key_value_heads,
                "hidden_act": hidden_act,
                "max_position_embeddings": max_position_embeddings,
                "initializer_range": initializer_range,
                "rms_norm_eps": rms_norm_eps,
                "use_cache": use_cache,
                "pad_token_id": pad_token_id,
                "bos_token_id": bos_token_id,
                "eos_token_id": eos_token_id,
                "tie_word_embeddings": tie_word_embeddings,
                "rope_theta": rope_theta,
                "attention_dropout": attention_dropout,
                "attention_bias": attention_bias,
            }
            logger.info("text_config is None. initializing the text model with default values.")
        text_model_type = text_config["model_type"] if "model_type" in text_config else "llama"
        self.text_config = CONFIG_MAPPING[text_model_type](**text_config)

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.rope_scaling = rope_scaling
        self.text_pretrained_model_name_or_path=text_pretrained_model_name_or_path
        self.image_encoder = image_encoder
        self._rope_scaling_validation()

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    # Copied from transformers.models.llama.configuration_llama.LlamaConfig._rope_scaling_validation
    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `type` and `factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}")


class DragonflyForCausalLM(FuyuPreTrainedModel):
    """
    Dragonfly class

    Args:
        config: DragonflyConfig
    """

    def __init__(self, config: DragonflyConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.language_model = AutoModelForCausalLM.from_config(config.text_config)
        self.use_image_encoder = config.image_encoder is not None

        if config.image_encoder is not None:
            print("Initialize Vision Encoder")
            self.image_encoder = CLIPVisionModel.from_pretrained(config.image_encoder)
            self.img_enc_dim = self.image_encoder.config.hidden_size

        if config.image_encoder is not None:
            self.vision_embed_tokens = nn.Linear(self.image_encoder.config.hidden_size, config.hidden_size)
        else:
            self.vision_embed_tokens = nn.Linear(config.patch_size * config.patch_size * config.num_channels, config.hidden_size)

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)
    
    def get_language_model(self):
        return self.language_model
    
    def initialize_model(self):
        if hasattr(self.config, "text_pretrained_model_name_or_path"):
            if self.config.text_pretrained_model_name_or_path is not None:
                language_model_id = self.config.text_pretrained_model_name_or_path
                self.language_model = AutoModelForCausalLM.from_pretrained(language_model_id, use_flash_attention_2=True)
        
        if hasattr(self.config, "image_encoder"):
            if self.config.image_encoder is not None:
                print("Initialize Vision Encoder!!")
                image_encoder_id = self.config.image_encoder
                self.image_encoder = CLIPVisionModel.from_pretrained(image_encoder_id)
                self.img_enc_dim = self.image_encoder.config.hidden_size
                self.vision_embed_tokens = nn.Linear(self.image_encoder.config.hidden_size, self.config.hidden_size)

    def gather_continuous_embeddings(
        self,
        word_embeddings: torch.Tensor,
        continuous_embeddings: List[torch.Tensor],
        image_patch_input_indices: torch.Tensor,
    ) -> torch.Tensor:
        """This function places the continuous_embeddings into the word_embeddings at the locations
        indicated by image_patch_input_indices. Different batch elements can have different numbers of continuous
        embeddings.

        Args:
            word_embeddings: Tensor of word embeddings. Shape: [b, s, h]
            continuous_embeddings:
                Tensor of continuous embeddings. The length of the list is the batch size. Each entry is
            shape [num_image_embeddings, hidden], and num_image_embeddings needs to match the number of non-negative
            indices in image_patch_input_indices for that batch element.
            image_patch_input_indices: Tensor of indices of the image patches in the input_ids tensor. Shape: [b, s]
        """
        if not (word_embeddings.shape[0] == len(continuous_embeddings)):
            raise ValueError(f"Batch sizes must match! Got {len(continuous_embeddings)=} and {word_embeddings.shape[0]=}")

        output_embeddings = word_embeddings.clone()
        for batch_idx in range(word_embeddings.shape[0]):
            # First, find the positions of all the non-negative values in image_patch_input_indices, those are the
            # positions in word_embeddings that we want to replace with content from continuous_embeddings.
            dst_indices = torch.nonzero(image_patch_input_indices[batch_idx] >= 0, as_tuple=True)[0]
            # Next look up those indices in image_patch_input_indices to find the indices in continuous_embeddings that we
            # want to use to replace the values in word_embeddings.
            src_indices = image_patch_input_indices[batch_idx][dst_indices]
            # Check if we have more indices than embeddings. Note that we could have fewer indices if images got truncated.
            if src_indices.shape[0] > continuous_embeddings[batch_idx].shape[0]:
                raise ValueError(f"Number of continuous embeddings {continuous_embeddings[batch_idx].shape=} does not match " f"number of continuous token ids {src_indices.shape=} in batch element {batch_idx}.")
            output_embeddings[batch_idx, dst_indices] = continuous_embeddings[batch_idx][src_indices]
        return output_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        labels: torch.LongTensor = None,
        image_patches: List[torch.Tensor] = None,
        image_patches_indices: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        topk: Optional[int] = 3,
        region_token_interval: Optional[int] = 6
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
            if image_patches is not None and past_key_values is None:

                query_hd_patches = [full_img_patches[5:] for full_img_patches in image_patches]
                query_outputs = [
                    self.image_encoder(
                        patch_pixel_values.to(self.vision_embed_tokens.weight.dtype), output_hidden_states=True
                    ) for patch_pixel_values in query_hd_patches
                ]
                query_image_patches = [item.hidden_states[-2] for item in query_outputs]
                query_image_patches = [self.vision_embed_tokens(patch.to(self.vision_embed_tokens.weight.dtype)) for patch in query_image_patches]
                query_ranks = [torch.mean(query_item,1) for query_item in query_image_patches]
                image_patches = [full_img_patches[:5] for full_img_patches in image_patches]
                ie_outputs = [
                    self.image_encoder(
                        patch_pixel_values.to(self.vision_embed_tokens.weight.dtype), output_hidden_states=True
                    ).hidden_states[-2] for patch_pixel_values in image_patches
                ]
                ie_outputs = [self.vision_embed_tokens(patch_pixel_values.to(self.vision_embed_tokens.weight.dtype)) for patch_pixel_values in ie_outputs]
                # region 1
                abstract_query1 = [torch.mean(patch_item[1],0) for patch_item in ie_outputs]
                query_ranks1 =[torch.matmul(qr, abstract.unsqueeze(-1)).squeeze(-1) for qr, abstract in zip(query_ranks, abstract_query1)]
                query_ranks_mask1 = torch.zeros(query_ranks[0].size()[:-1]).to(query_ranks[0].device)
                query_ranks_mask1.scatter_(0, torch.arange(region_token_interval,region_token_interval*4).to(query_ranks[0].device), float('-inf'))
                query_ranks1 = [T + query_ranks_mask1 for T in query_ranks1]
                query_ranks1 = [torch.topk(item, topk).indices for item in query_ranks1]
                # region 2
                abstract_query2 = [torch.mean(patch_item[2],0) for patch_item in ie_outputs]
                query_ranks2 = [torch.matmul(qr, abstract.unsqueeze(-1)).squeeze(-1) for qr, abstract in zip(query_ranks, abstract_query2)]
                query_ranks_mask2 = torch.zeros(query_ranks[0].size()[:-1]).to(query_ranks[0].device)
                query_ranks_mask2.scatter_(0, torch.concat([torch.arange(0,region_token_interval).to(query_ranks[0].device), torch.arange(region_token_interval*2,region_token_interval*4).to(query_ranks[0].device)]), float('-inf'))
                query_ranks2 = [T + query_ranks_mask2 for T in query_ranks2]
                query_ranks2 = [torch.topk(item, topk).indices for item in query_ranks2]
                # region 3
                abstract_query3 = [torch.mean(patch_item[3],0) for patch_item in ie_outputs]
                query_ranks3 =[torch.matmul(qr, abstract.unsqueeze(-1)).squeeze(-1) for qr, abstract in zip(query_ranks, abstract_query3)]
                query_ranks_mask3 = torch.zeros(query_ranks[0].size()[:-1]).to(query_ranks[0].device) # add
                query_ranks_mask3.scatter_(0, torch.concat([torch.arange(0,region_token_interval*2).to(query_ranks[0].device), torch.arange(region_token_interval*3,region_token_interval*4).to(query_ranks[0].device)]), float('-inf'))  #add
                query_ranks3 = [T + query_ranks_mask3 for T in query_ranks3] #add
                query_ranks3 = [torch.topk(item, topk).indices for item in query_ranks3]
                # region 4
                abstract_query4 = [torch.mean(patch_item[4],0) for patch_item in ie_outputs]
                query_ranks4 =[torch.matmul(qr, abstract.unsqueeze(-1)).squeeze(-1) for qr, abstract in zip(query_ranks, abstract_query4)]
                query_ranks_mask4 = torch.zeros(query_ranks[0].size()[:-1]).to(query_ranks[0].device) # add
                query_ranks_mask4.scatter_(0, torch.arange(0,region_token_interval*3).to(query_ranks[0].device), float('-inf'))  #add
                query_ranks4 = [T + query_ranks_mask4 for T in query_ranks4] #add
                query_ranks4 = [torch.topk(item, topk).indices for item in query_ranks4]
                # Construct visual encoding
                query_ranks = [torch.concat([q1,q2,q3,q4]) for q1,q2,q3,q4 in zip(query_ranks1,query_ranks2,query_ranks3,query_ranks4)]
                query_ranks = [torch.sort(item).values for item in query_ranks]
                selected_image_patches = [
                    torch.index_select(q_image_patches, 0, q_ranks)
                    for (q_image_patches, q_ranks) in zip(query_image_patches, query_ranks) 
                ]

                # concat
                patch_embeddings = [
                    torch.concat([
                        ie_output.view(-1, self.config.hidden_size),
                        s_image_patches.view(-1, self.config.hidden_size)
                    ]) for ie_output, s_image_patches in zip(ie_outputs, selected_image_patches)
                ]

                inputs_embeds = self.gather_continuous_embeddings(
                    word_embeddings=inputs_embeds,
                    continuous_embeddings=patch_embeddings,
                    image_patch_input_indices=image_patches_indices,
                )

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            labels=labels,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            output_hidden_states=True
        )

        if not return_dict:
            return tuple(v for v in outputs if v is not None)
        return outputs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        image_patches=None,
        image_patches_indices=None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        if image_patches is not None and image_patches_indices is not None:
            model_inputs["image_patches_indices"] = image_patches_indices
            model_inputs["image_patches"] = image_patches

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "image_patches_indices": image_patches_indices if past_key_values is None else None,
                "image_patches": image_patches if past_key_values is None else None,
            }
        )
        return model_inputs
