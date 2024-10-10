import base64
import io
import json
import os
import random
import re
import sys
import urllib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from multiprocessing import Value

import numpy as np
import torch
import torch.utils
from datasets import interleave_datasets, load_dataset
from datasets.distributed import split_dataset_by_node
from PIL import Image

from src.dragonfly.models.processing_dragonfly import DragonflyProcessor

sys.path.append("../..")
import json
import os

import yaml
from datasets.utils.file_utils import get_datasets_user_agent
from PIL import Image, ImageFile

from pipeline.train.train_utils import DistributedProxySampler

USER_AGENT = get_datasets_user_agent()

Image.MAX_IMAGE_PIXELS = 1000000000

IMAGE_CAP_INSTRUCT = [
    "Analyze the image in a comprehensive and detailed manner.",
    "What's happening in the scene?",
    "Write a terse but informative summary of the picture.",
    "What are the key elements in this picture?",
    "Present a compact description of the photo's key features.",
    "What do you think is going on in this snapshot?",
    "Describe the following image.",
    "What do you see happening in this image?",
    "Provide a brief description of the given image.",
    "What is this photo about'?",
    "Summarize the visual content of the image.",
    "What is in the photo?",
    "Write a detailed description of the given image.",
    "Can you elaborate on the elements of the picture provided?",
    "Give a brief description of the image.",
    "Explain the visual content of the image in great detail.",
    "Render a clear and concise summary of the photo.",
    "Describe the image concisely.",
    "Give a short and clear explanation of the subsequent image.",
    "Can you describe the main features of this image for me?",
    "Share a concise interpretation of the image provided.",
    "What is this?",
]


def format_llama3_prompt(item):
    if "text" in item:
        instruction = random.choice(IMAGE_CAP_INSTRUCT)
        formated_prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{item['text']}<|eot_id|>"
    else:
        formated_prompt = item["conversations"]
    return formated_prompt


def prepare_dragonfly_sample(batch_data, processor, image_dir=None, max_length=None):
    def hq_dataset_pp(example):
        if "image_url" in example and example["image_url"] is not None and example["image_url"].strip() != "":
            if example["image_url"].startswith("/data/"):
                image_url = example["image_url"]
            else:
                image_url = os.path.join(image_dir, example["image_url"])
            img = Image.open(image_url).convert("RGB")
        else:
            img = None
        return img

    for item in batch_data:
        item["text"] = format_llama3_prompt(item)
        if "image" not in item:
            item["image"] = hq_dataset_pp(item)

    pil_images = [item["image"] for item in batch_data]
    if None in pil_images:
        pil_images = None

    texts = [item["text"] for item in batch_data]
    model_inputs = processor(text=texts, images=pil_images, return_tensors="pt", truncation=True, max_length=max_length)
    labels = processor.get_labels(input_ids=model_inputs["input_ids"], special_token_id=128000)
    input_ids, labels = processor.find_and_remove_tokens(input_ids=model_inputs["input_ids"], labels=labels, token_id=128000)
    model_inputs["input_ids"] = input_ids
    model_inputs["labels"] = labels
    return model_inputs


def load_dragonfly_val_dataset(args):
    processor = args.processor
    data_files = args.val_files.split(",")

    def hq_dataset_pp(example):
        imgs = []
        for image_url in example["image_url"]:
            image_url = os.path.join(args.image_dir, image_url)
            imgs.append(Image.open(image_url).convert("RGB"))
        example["image"] = imgs
        return example

    val_dataset = load_dataset("parquet", data_files=data_files, split="train", cache_dir=args.data_cache_dir, streaming=True)
    val_dataset = split_dataset_by_node(val_dataset, world_size=args.world_size, rank=args.rank)
    val_dataset = val_dataset.map(hq_dataset_pp, batched=True, batch_size=args.batch_size)
    val_dataset = val_dataset.remove_columns(["image_url", "source"])

    return val_dataset


def load_dragonfly_pretrain_dataset(args):
    processor = args.processor
    data_dir = args.data_dir
    data_files = [f for f in os.listdir(data_dir) if f.endswith(".parquet")]
    hq_data_files = []
    lq_data_files = []
    text_data_files = []
    math_data_files = []

    together_hq_datasets = args.together_hq_datasets.split(",") if args.together_hq_datasets else []
    together_lq_datasets = args.together_lq_datasets.split(",") if args.together_lq_datasets else []
    together_text_datasets = args.together_text_datasets.split(",") if args.together_text_datasets else []
    together_math_datasets = args.together_math_datasets.split(",") if args.together_math_datasets else []
    print(together_hq_datasets)
    print(together_text_datasets)
    print(together_math_datasets)
    for fpath in data_files:
        for d in together_hq_datasets:
            if fpath.startswith(d):
                hq_data_files.append(os.path.join(data_dir, fpath))
                break
        for d in together_lq_datasets:
            if fpath.startswith(d):
                lq_data_files.append(os.path.join(data_dir, fpath))
                break
        for d in together_text_datasets:
            if fpath.startswith(d):
                text_data_files.append(os.path.join(data_dir, fpath))
                break
        for d in together_math_datasets:
            if fpath.startswith(d):
                math_data_files.append(os.path.join(data_dir, fpath))
                break
    print(hq_data_files)
    print(text_data_files)
    print(math_data_files)

    hq_dataset = load_dataset("parquet", data_files=hq_data_files, split="train", cache_dir=args.data_cache_dir)
    hq_dataset = split_dataset_by_node(hq_dataset, world_size=args.world_size, rank=args.rank)
    hq_dataset = hq_dataset.remove_columns(["source"])
    dataset = hq_dataset

    dataset = dataset.shuffle(seed=args.seed)

    if text_data_files:
        text_dataset = load_dataset("parquet", data_files=text_data_files, split="train", cache_dir=args.data_cache_dir, streaming=True)
        text_dataset = split_dataset_by_node(text_dataset, world_size=args.world_size, rank=args.rank)
        # text_dataset = text_dataset.remove_columns(["source"])
        text_dataset = text_dataset.shuffle(seed=args.seed, buffer_size=1000)
    else:
        text_dataset = None

    if math_data_files:
        math_dataset = load_dataset("parquet", data_files=math_data_files, split="train", cache_dir=args.data_cache_dir, streaming=True)
        math_dataset = split_dataset_by_node(math_dataset, world_size=args.world_size, rank=args.rank)
        math_dataset = math_dataset.shuffle(seed=args.seed, buffer_size=1000)
    else:
        math_dataset = None

    return dataset, text_dataset, math_dataset
