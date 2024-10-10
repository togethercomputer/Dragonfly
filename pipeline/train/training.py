""" Main training script """

import argparse
import gc
import glob
import os
import random
import shutil
import sys
import time
from functools import partial
from itertools import cycle

import numpy as np
import torch
import torch.nn
import wandb
from accelerate import Accelerator, load_checkpoint_and_dispatch
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

sys.path.append("../..")
from pipeline.data_utils.data import (
    load_dragonfly_pretrain_dataset,
    load_dragonfly_val_dataset,
    prepare_dragonfly_sample,
)
from pipeline.train.distributed import world_info_from_env
from pipeline.train.train_utils import (
    AverageMeter,
    delete_tensors_from_dict,
    get_next_dataloader,
    random_seed,
    save_checkpoint_weights,
    save_final_weights,
)
from src.dragonfly.models.modeling_dragonfly import (
    DragonflyConfig,
    DragonflyForCausalLM,
)
from src.dragonfly.models.processing_dragonfly import (
    IMAGE_NEWLINE_TOKEN,
    IMAGE_PLACEHOLDER_TOKEN,
    DragonflyProcessor,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)


def get_grouped_params(model, wd):
    params_with_wd, params_without_wd = [], []

    def apply_decay(x):
        return "weight" in x and "layernorm" not in x and "lm_head" not in x and "embed_tokens" not in x

    for n, p in model.named_parameters():
        if p.requires_grad:
            if apply_decay(n):
                params_with_wd.append(p)
            else:
                params_without_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": wd},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--external_save_dir",
        type=str,
        default=None,
        help="set to save model to external path",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        help="Whether to resume from checkpoint, if set True, will load models from --external_save_dir",
    )
    parser.add_argument(
        "--delete_previous_checkpoint",
        action="store_true",
        help="delete previous checkpoint when saving new checkpoint",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="dragonfly",
        help="used to name saving directory and wandb run",
    )
    parser.add_argument(
        "--data_cache_dir",
        type=str,
        default="/scratch/.hf_cache/datasets",
        help="used to name saving directory and wandb run",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="path to dataset",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        help="path to dataset",
    )
    parser.add_argument(
        "--together_hq_datasets",
        type=str,
        default="cc3m,llava_pretrain,shareGPT4V",
        help="high quality dataset with images saved locally",
    )
    parser.add_argument(
        "--together_lq_datasets",
        type=str,
        default="",
        help="low quality dataset that requires image download",
    )
    parser.add_argument(
        "--together_text_datasets",
        type=str,
        default="",
        help="text-only dataset",
    )
    parser.add_argument(
        "--together_math_datasets",
        type=str,
        default="",
        help="text-only dataset",
    )
    parser.add_argument(
        "--hq_dataset_prob",
        type=float,
        default=1.0,
        help="hq and lq dataset weights",
    )
    parser.add_argument(
        "--val_files",
        type=str,
        default="",
        help="validation dataset",
    )
    parser.add_argument(
        "--text_alternate_step",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--math_alternate_step",
        type=int,
        default=50,
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dataset_resampled", action="store_true")
    # parser.add_argument("--use_media_placement_augmentation", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--total_training_steps", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=100, help="log loss every n steps")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=10000,
        help="checkpointing every n steps",
    )
    parser.add_argument(
        "--total_checkpoint_limits",
        type=int,
        default=3,
        help="Total checkpoints to save",
    )
    parser.add_argument("--save_hf_checkpoints", action="store_true", help="Saving huggingface style checkpoints")
    parser.add_argument(
        "--total_hf_checkpoint_limits",
        type=int,
        default=5,
        help="Total checkpoints to save",
    )
    parser.add_argument(
        "--hf_checkpointing_steps",
        type=int,
        default=160000,
        help="checkpointing every n steps",
    )
    # Sum of gradient optimization batch size

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        help="path to huggingface model or model identifier from local path or huggingface.co",
        default=None,
    )
    parser.add_argument(
        "--text_pretrained_model_name_or_path",
        type=str,
        help="path to text model",
        default="teknium/OpenHermes-2.5-Mistral-7B",
    )
    parser.add_argument(
        "--image_encoder_name_or_path",
        type=str,
        default=None,
        help="Whether to use image encoder.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument(
        "--lr_scheduler",
        default="constant",
        type=str,
        help="constant, linear, or cosine",
    )
    parser.add_argument("--mm_tune_vision_encoder", default=False, action="store_true")
    parser.add_argument("--tune_vision_embed_tokens_only", default=False, action="store_true")
    parser.add_argument("--loss_multiplier_ce", type=float, default=1.0)
    parser.add_argument("--loss_multiplier_ic", type=float, default=0.0)
    parser.add_argument("--warmup_steps", default=1000, type=int)
    parser.add_argument("--warmup_steps_ratio", default=None, type=float)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    # distributed training args
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )
    # YH: Training detail
    parser.add_argument("--mask_lm_head", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=4096,
        help="the maximum sequence length",
    )
    parser.add_argument("--patch-image-size", type=int, default=224)
    # this could potentially save 33GB of all model parameters for otter-9b, including the language and vision model.
    parser.add_argument("--save_hf_model", default=False, action="store_true")
    # wandb args
    parser.add_argument("--report_to_wandb", default=False, action="store_true")
    parser.add_argument(
        "--wandb_project",
        type=str,
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
    )
    parser.add_argument(
        "--save_checkpoints_to_wandb",
        default=False,
        action="store_true",
        help="save checkpoints to wandb",
    )
    return parser


def train_one_epoch(
    args,
    model,
    epoch,
    dataloader,
    text_dataloader,
    math_dataloader,
    val_dataloader,
    tokenizer,
    optimizer,
    processor,
    lr_scheduler,
    device_id,
    accelerator,
    current_global_steps,
    wandb,
):

    total_training_steps = args.total_training_steps
    if text_dataloader is not None:
        alternate_step = args.text_alternate_step
        text_dataloader = cycle(text_dataloader)

    if math_dataloader is not None:
        math_alternate_step = args.math_alternate_step
        math_dataloader = cycle(math_dataloader)

    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)["input_ids"][-1]

    model.train()

    # setup logging
    step_time_m = AverageMeter()  # time for one optimizer step (> 1 batch if using gradient accum)
    data_time_m = AverageMeter()  # avg time to load one batch of both C4 AND cc3m (= 1 batch regardless of gradient accum)
    end = time.time()
    dtype = model.dtype
    print(f"Using dtype {dtype}")

    # loop through dataloader
    for num_steps, batch in tqdm(
        enumerate(dataloader),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=current_global_steps,
    ):

        data_time_m.update(time.time() - end)
        global_step = num_steps + current_global_steps

        if global_step >= total_training_steps:
            break

        if global_step <= 1:
            # for k, v in batch.items():
            #     print(v)
            print(batch["input_ids"].tolist())
            print(batch["input_ids"].size())

        #### FORWARD PASS ####
        with accelerator.accumulate(model):
            model_inputs = {}
            for k, v in batch.items():
                if v is None:
                    model_inputs[k] = None
                elif isinstance(v, torch.Tensor):
                    model_inputs[k] = v.to("cuda")
                else:
                    model_inputs[k] = [vv.to("cuda") for vv in v]

            with accelerator.autocast():
                model_outputs = model(**model_inputs)
                total_loss = model_outputs["loss"]

            #### BACKWARD ####
            accelerator.backward(total_loss)

            text_batch = None
            if text_dataloader is not None and (global_step + 1) % alternate_step == 0:
                text_batch = next(text_dataloader)
                if global_step < 10:
                    print(text_batch["input_ids"].tolist())
                model_text_inputs = {}
                for k, v in text_batch.items():
                    if v is None:
                        model_text_inputs[k] = None
                    elif isinstance(v, torch.Tensor):
                        model_text_inputs[k] = v.to("cuda")
                    else:
                        model_text_inputs[k] = [vv.to("cuda") for vv in v]

                with accelerator.autocast():
                    model_text_outputs = model(**model_text_inputs)
                    text_total_loss = model_text_outputs["loss"]

                #### BACKWARD ####
                accelerator.backward(text_total_loss)

                total_loss = (total_loss + text_total_loss) / 2.0
                delete_tensors_from_dict(text_batch)

            math_batch = None
            if math_dataloader is not None and (global_step + 1) % math_alternate_step == 0:
                math_batch = next(math_dataloader)
                if global_step < 40:
                    print(math_batch["input_ids"].tolist())
                model_math_inputs = {}
                for k, v in math_batch.items():
                    if v is None:
                        model_math_inputs[k] = None
                    elif isinstance(v, torch.Tensor):
                        model_math_inputs[k] = v.to("cuda")
                    else:
                        model_math_inputs[k] = [vv.to("cuda") for vv in v]

                with accelerator.autocast():
                    model_math_outputs = model(**model_math_inputs)
                    math_total_loss = model_math_outputs["loss"]

                #### BACKWARD ####
                accelerator.backward(math_total_loss)

                total_loss = (total_loss + math_total_loss) / 2.0

                delete_tensors_from_dict(math_batch)

            def mask_embedding(m):
                if m.weight.requires_grad:
                    zero_mask = torch.zeros_like(m.weight.grad)
                    zero_mask[media_token_id] = torch.ones_like(zero_mask[media_token_id])
                    zero_mask[endofchunk_token_id] = torch.ones_like(zero_mask[endofchunk_token_id])
                    m.weight.grad = m.weight.grad * zero_mask

            if args.mask_lm_head and args.distributed_type != "DEEPSPEED":
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.language_model.model.embed_tokens.apply(mask_embedding)
                unwrapped_model.language_model.lm_head.apply(mask_embedding)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # print(f"Step 3: Beginning Step: {num_steps}; Global Step: {global_step}")

            # step time and reset end outside of rank 0
            step_time_m.update(time.time() - end)
            end = time.time()

            if accelerator.sync_gradients:
                if args.rank == 0 and args.report_to_wandb:
                    # compute within rank 0
                    samples_per_second = args.gradient_accumulation_steps * args.batch_size * args.world_size / step_time_m.val
                    samples_per_second_per_gpu = args.gradient_accumulation_steps * args.batch_size / step_time_m.val
                    wandb.log(
                        {
                            "data_time": data_time_m.avg,
                            "step_time": step_time_m.avg,
                            "samples_per_second": samples_per_second,
                            "samples_per_second_per_gpu": samples_per_second_per_gpu,
                            "lr": optimizer.param_groups[0]["lr"],
                        },
                        commit=False,
                    )
                    step_time_m.reset()
                    data_time_m.reset()

                    wandb.log(
                        {
                            "loss": total_loss.item(),
                            "global_step": global_step,
                        },
                        commit=True,
                    )

        delete_tensors_from_dict(batch)
        if text_batch is not None:
            delete_tensors_from_dict(text_batch)

        # Log loss to console
        if (global_step % args.logging_steps == 0) and args.rank == 0:
            print(f"Step {global_step} of total {total_training_steps} steps complete. Loss: {total_loss.item():.3f}.")

        if global_step % args.logging_steps == 0:
            batch = None
            text_batch = None
            gc.collect()
            torch.cuda.empty_cache()

        if global_step != 0 and global_step % args.logging_steps == 0:
            if val_dataloader is not None:
                model.eval()
                with torch.no_grad():
                    num_val_batch = 0

                    for val_num_steps, val_batch in tqdm(enumerate(val_dataloader)):
                        val_model_inputs = {}
                        for k, v in val_batch.items():
                            val_model_inputs[k] = v.to(device_id, non_blocking=True) if isinstance(v, torch.Tensor) else [vv.to(device_id, non_blocking=True) for vv in v]
                        with accelerator.autocast():
                            val_model_outputs = model(**model_inputs)
                            num_val_batch += 1
                    val_batch = None
                gc.collect()
                torch.cuda.empty_cache()
                model.train()

        # Add a process on saving checkpoints during pretraining
        if global_step != 0 and (global_step % args.checkpointing_steps == 0):
            if accelerator.is_main_process and not os.path.exists(args.external_save_dir):
                os.makedirs(args.external_save_dir)

            print(f"Saving checkpoint to {args.external_save_dir}/checkpoint_steps_{epoch}_{global_step}")
            checkpoint_dir = f"{args.external_save_dir}/checkpoint_steps_{epoch}_{global_step}"
            accelerator.save_state(checkpoint_dir)

            checkpoint_list = glob.glob(f"{args.external_save_dir}/checkpoint_steps_*")
            if args.rank == 0 and len(checkpoint_list) > args.total_checkpoint_limits:
                delete_checkpoint_path = sorted(
                    checkpoint_list,
                    key=lambda x: int(x.split("_")[-1]),
                )[0]
                print(f"Deleting checkpoint to {args.external_save_dir}/{delete_checkpoint_path}")
                if os.path.exists(delete_checkpoint_path):
                    shutil.rmtree(delete_checkpoint_path)
        try:
            if args.save_hf_checkpoints and global_step != 0 and (global_step % args.hf_checkpointing_steps == 0):
                if accelerator.is_main_process and not os.path.exists(args.external_save_dir):
                    os.makedirs(args.external_save_dir, exist_ok=True)

                checkpoint_dir = f"{args.external_save_dir}/hf_checkpoint_steps_{epoch}_{global_step}"
                print(f"Saving huggingface checkpoint to {checkpoint_dir}")

                accelerator.wait_for_everyone()
                save_checkpoint_weights(model, checkpoint_dir, accelerator, processor=processor, tokenizer=tokenizer)

                checkpoint_list = glob.glob(f"{args.external_save_dir}/hf_checkpoint_steps_*")
                if args.rank == 0 and len(checkpoint_list) > args.total_hf_checkpoint_limits:
                    delete_checkpoint_path = sorted(
                        checkpoint_list,
                        key=lambda x: int(x.split("_")[-1]),
                    )[0]
                    print(f"Deleting checkpoint to {args.external_save_dir}/{delete_checkpoint_path}")
                    if os.path.exists(delete_checkpoint_path):
                        shutil.rmtree(delete_checkpoint_path)
        except:
            print("Save HF checkpoint Error.")

    return global_step


def main():
    parser = parse_args()
    # TODO: remove additional data args, all args would be processed in above parser
    # parser = add_data_args(parser)
    args = parser.parse_args()

    if args.save_checkpoints_to_wandb and not args.report_to_wandb:
        raise ValueError("save_checkpoints_to_wandb requires report_to_wandb")

    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    args.local_rank, args.rank, args.world_size = world_info_from_env()
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    if accelerator.state.deepspeed_plugin is not None:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = args.batch_size

    device_id = accelerator.device

    random_seed(args.seed)

    if args.pretrained_model_name_or_path is not None:
        accelerator.print(f"Loading pretrained moel from {args.pretrained_model_name_or_path}")
        device_map = {"": device_id} if accelerator.distributed_type == "MULTI_GPU" or accelerator.distributed_type == "DEEPSPEED" else "auto"
        kwargs = {"local_files_only": args.offline}
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)
        tokenizer.eos_token = "<|end_of_text|>"
        tokenizer.add_bos_token = True
        tokenizer.add_eos_token = True
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token
        if args.image_encoder_name_or_path is not None:
            clip_processor = AutoProcessor.from_pretrained(args.image_encoder_name_or_path)
            image_processor = clip_processor.image_processor
        else:
            image_processor = None
        assert image_processor is not None
        processor = DragonflyProcessor(image_processor=image_processor, tokenizer=tokenizer, image_encoding_style="llava-hd")
        model = DragonflyForCausalLM.from_pretrained(args.pretrained_model_name_or_path, **kwargs)
        args.processor = processor
    else:
        accelerator.print(f"Initialize model from scratch")

        tokenizer = AutoTokenizer.from_pretrained(args.text_pretrained_model_name_or_path)
        tokenizer.eos_token = "<|end_of_text|>"
        tokenizer.add_eos_token = True
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token
        if args.image_encoder_name_or_path is not None:
            clip_processor = AutoProcessor.from_pretrained(args.image_encoder_name_or_path)
            image_processor = clip_processor.image_processor
        else:
            image_processor = None
        assert image_processor is not None
        processor = DragonflyProcessor(image_processor=image_processor, tokenizer=tokenizer, image_encoding_style="llava-hd")
        model_config = DragonflyConfig(
            text_pretrained_model_name_or_path=args.text_pretrained_model_name_or_path,
            image_encoder=args.image_encoder_name_or_path,
        )
        model = DragonflyForCausalLM(model_config)
        args.processor = processor
        model.initialize_model()

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if args.mm_tune_vision_encoder:
        for p in model.language_model.parameters():
            p.requires_grad = False
        if args.tune_vision_embed_tokens_only:
            for p in model.image_encoder.parameters():
                p.requires_grad = False
            print("Vision_embed_tokens grad true")
            for p in model.vision_embed_tokens.parameters():
                p.requires_grad = True

    args.tokenizer = tokenizer

    accelerator.wait_for_everyone()

    args.distributed_type = accelerator.distributed_type

    random_seed(args.seed, args.rank)

    dataset, text_dataset, math_dataset = load_dragonfly_pretrain_dataset(args)

    if args.val_files:
        val_dataset = load_dragonfly_val_dataset(args)

    else:
        val_dataset = None

    total_training_steps = args.total_training_steps

    optimizer = torch.optim.AdamW(get_grouped_params(model, wd=args.weight_decay), lr=args.learning_rate)

    if args.rank == 0:
        print(f"Total training steps: {total_training_steps}")

    args.warmup_steps = total_training_steps * args.warmup_steps_ratio if args.warmup_steps_ratio is not None else args.warmup_steps

    if args.lr_scheduler == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps // args.gradient_accumulation_steps,
            num_training_steps=total_training_steps // args.gradient_accumulation_steps,
        )
    elif args.lr_scheduler == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps // args.gradient_accumulation_steps,
            num_training_steps=total_training_steps // args.gradient_accumulation_steps,
        )
    else:
        lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)

    if args.rank == 0 and args.report_to_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
        )

    accelerator.print("***** Running training *****")
    accelerator.print(f"  High Quality Datasets = {args.together_hq_datasets}")
    accelerator.print(f"  Low Quality Datasets = {args.together_lq_datasets}")
    accelerator.print(f"  Text Datasets = {args.together_text_datasets}")
    accelerator.print(f"  Math Datasets = {args.together_math_datasets}")
    accelerator.print(f"  Num steps = {args.total_training_steps}")
    accelerator.print(f"  Instantaneous batch size per device = {args.batch_size}")

    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    required_grad_ps = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            required_grad_ps.append(name)
    print("Parameters required grad: ")
    print(required_grad_ps)

    current_global_steps = 0
    resume_from_epoch = 0
    # check if a checkpoint exists for this run
    args.external_save_dir = os.path.join(args.external_save_dir, args.run_name) if args.external_save_dir else args.run_name
    if os.path.exists(f"{args.external_save_dir}") and args.resume_from_checkpoint is True:
        checkpoint_list = glob.glob(f"{args.external_save_dir}/checkpoint_steps_*")
        if len(checkpoint_list) == 0:
            print(f"Found no checkpoints for run {args.external_save_dir}.")
        else:
            resume_from_checkpoint_path = sorted(
                checkpoint_list,
                key=lambda x: int(x.split("_")[-1]),  # steps_<epoch>_<steps>
            )[-1]
            # resume_from_checkpoint_path = sorted(checkpoint_list, key=lambda x: int(x.split("_")[-1].split(".")[0]))[-1]
            print(f"Found checkpoint {resume_from_checkpoint_path} for run {args.external_save_dir}.")

            if args.rank == 0:
                print(f"Loading checkpoint from {resume_from_checkpoint_path}")
            accelerator.load_state(resume_from_checkpoint_path)
            current_global_steps = int(resume_from_checkpoint_path.split("_")[-1]) + 1
            resume_from_epoch = int(resume_from_checkpoint_path.split("_")[-2])

    if accelerator.num_processes > 1:
        lr_scheduler.split_batches = True

    model.train()

    epoch = resume_from_epoch

    while True:
        if current_global_steps >= args.total_training_steps:
            break

        # dataset.set_epoch(epoch)
        dataset = dataset.shuffle(seed=args.seed + epoch)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=partial(
                prepare_dragonfly_sample,
                image_dir=args.image_dir,
                processor=processor,
                max_length=args.max_seq_length,
            ),
        )
        if text_dataset is not None:
            text_dataset.set_epoch(epoch)
            text_dataloader = torch.utils.data.DataLoader(
                text_dataset,
                batch_size=args.batch_size,
                num_workers=args.workers,
                pin_memory=True,
                drop_last=True,
                collate_fn=partial(
                    prepare_dragonfly_sample,
                    image_dir=args.image_dir,
                    processor=processor,
                    max_length=args.max_seq_length,
                ),
            )
        else:
            text_dataloader = None

        if math_dataset is not None:
            math_dataset.set_epoch(epoch)
            math_dataloader = torch.utils.data.DataLoader(
                math_dataset,
                batch_size=args.batch_size,
                num_workers=args.workers,
                pin_memory=True,
                drop_last=True,
                collate_fn=partial(
                    prepare_dragonfly_sample,
                    image_dir=args.image_dir,
                    processor=processor,
                    max_length=args.max_seq_length,
                ),
            )
        else:
            math_dataloader = None

        if args.val_files:
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=1,
                num_workers=args.workers,
                pin_memory=True,
                drop_last=True,
                collate_fn=partial(
                    prepare_dragonfly_sample,
                    processor=processor,
                    max_length=args.max_seq_length,
                ),
            )
        else:
            val_dataloader = None

        current_global_steps = train_one_epoch(
            args=args,
            model=model,
            epoch=epoch,
            current_global_steps=current_global_steps,
            tokenizer=tokenizer,
            optimizer=optimizer,
            processor=processor,
            lr_scheduler=lr_scheduler,
            dataloader=dataloader,
            text_dataloader=text_dataloader,
            math_dataloader=math_dataloader,
            val_dataloader=val_dataloader,
            accelerator=accelerator,
            device_id=device_id,
            wandb=wandb,
        )
        accelerator.wait_for_everyone()
        epoch += 1

    accelerator.wait_for_everyone()
    save_final_weights(
        model,
        args,
        accelerator,
        processor=processor,
        tokenizer=tokenizer,
    )


if __name__ == "__main__":
    main()
