""" Testing script """

import sys
from dragonfly.models.modeling_dragonfly import *
from dragonfly.models.processing_dragonfly import *
from transformers import AutoProcessor, AutoTokenizer
from PIL import Image
import torch
from pipeline.train.train_utils import random_seed


def format_text(text, system_prompt=""):
    if len(system_prompt) > 0:
        instruction = f"{system_prompt} {text}"
    else:
        instruction = text
    prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    return prompt


# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)

# set your model name and image path
# pretrained_model_name_or_path = "togethercomputer/Llama-3-8B-Dragonfly-v1"
# image_path = "./test_images/skateboard.png"
# question = "Summarize the visual content of the image."

# For biomed
pretrained_model_name_or_path = "togethercomputer/togethercomputer/Llama-3-8B-Dragonfly-Med-v1"
image_path = "./test_images/ROCO_04197.jpg"
question = "Provide a brief description of the given image."

def main():

    random_seed(40)
    print(f"Loading pretrained model from {pretrained_model_name_or_path}")
    device_map = "cuda:0"
    kwargs = {}
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

    # load the image
    image = Image.open(image_path)
    image = image.convert('RGB')
    images = [image]

    # prepare inputs for the model
    text_prompt = format_text(question)

    # process the text and image
    inputs = processor(text=[text_prompt], images=[image], max_length=2048, return_tensors="pt", is_generate=True)
    inputs = inputs.to(f"cuda:0")


    # generate the response
    temperature = 0
    with torch.inference_mode():
        generation_output = model.generate(**inputs, 
                                            max_new_tokens=1024,
                                            eos_token_id=tokenizer.encode('<|eot_id|>'), 
                                            do_sample=temperature > 0, 
                                            temperature=temperature,
                                            use_cache=True)
    generation_text = processor.batch_decode(generation_output, skip_special_tokens=False)
    print(generation_text[0].replace('<|reserved_special_token_0|>','').replace('<|reserved_special_token_1|>',''))


if __name__ == "__main__":
    main()

