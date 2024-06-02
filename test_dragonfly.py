""" Main training script """

import argparse
import gc
import glob
import os
import shutil
import random
import sys
import time
from functools import partial
import re
import numpy as np
import torch
import torch.nn
from tqdm import tqdm
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    AutoTokenizer,
    FuyuImageProcessor,
    AutoProcessor
)
import wandb

from pipeline.train.train_utils import random_seed
from pipeline.train.distributed import world_info_from_env
from src.dragonfly.models.dragonfly.modeling_dragonfly import DragonflyForCausalLM, DragonflyConfig
from src.dragonfly.models.dragonfly.processing_dragonfly import (
    DragonflyProcessor,
    IMAGE_NEWLINE_TOKEN,
    IMAGE_PLACEHOLDER_TOKEN,
)
import os
import torch
import requests
from PIL import Image
from safetensors import safe_open
from datasets import load_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)

pretrained_model_name_or_path = "/data/kezhen/multi-modality/checkpoints/final_runs_v3/raccoon_zoom_select_mix10_sharegptpt"

def main():

    random_seed(40)
    
    print(f"Loading pretrained model from {pretrained_model_name_or_path}")
    device_map = "cuda:0"
    kwargs = {"local_files_only": True}
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    clip_processor = AutoProcessor.from_pretrained('openai/clip-vit-base-patch32')
    image_processor = clip_processor.image_processor
    processor = DragonflyProcessor(image_processor=image_processor, tokenizer=tokenizer, image_encoding_style='llava-hd')
    model = DragonflyForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        **kwargs
    )
    model = model.to(torch.bfloat16)

    model = model.to("cuda:0")

    # prepare inputs for the model
    text_prompt = "<|start_header_id|>user<|end_header_id|>\n\nSummarize the visual content of the image.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


    img_urls = [
        "https://huggingface.co/adept/fuyu-8b/resolve/main/skateboard.png",
    ]
    for url in img_urls:
        if url.startswith('http'):
            image = Image.open(requests.get(url, stream=True).raw)
        else:
            image = Image.open(url).convert('RGB')

        inputs = processor(text=[text_prompt], images=[image], max_length=2048, return_tensors="pt", is_generate=True)
        print(inputs['input_ids'].size())
        inputs = inputs.to("cuda:0")

    
    # autoregressively generate text
        with torch.inference_mode():
            generation_output = model.generate(**inputs, max_new_tokens=1024,eos_token_id=tokenizer.encode('<|eot_id|>'), do_sample=True, temperature=0.2,use_cache=True)
        generation_text = processor.batch_decode(generation_output, skip_special_tokens=False)
        print(generation_text[0].replace('<|reserved_special_token_0|>','').replace('<|reserved_special_token_1|>',''))


if __name__ == "__main__":
    main()

