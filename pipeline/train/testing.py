import sys
from dragonfly.models.modeling_dragonfly import *
from dragonfly.models.processing_dragonfly import *
from transformers import AutoProcessor, AutoTokenizer
from PIL import Image
import torch


def format_text(text, system_prompt=""):
    if len(system_prompt) > 0:
        instruction = f"{system_prompt} {text}"
    else:
        instruction = text
    prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    return prompt

# for general domain
model_name = "togethercomputer/Dragonfly-v1-llama8b"
model_type = "llava-hd"
question = "What do you see in the image?"
image_path = "./test_images/skateboard.png"

# for biomedical domain
# model_name = "togethercomputer/Dragonfly-med-v1-llama8b"
# model_type = "llava-hd"
# question = "are the lungs normal appearing?"
# image_path = "./test_images/chext-xray.jpeg"


device = 0
temperature = 0.2
max_new_tokens = 64


# instantiate the tokenizer, processor, and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
clip_processor = AutoProcessor.from_pretrained('openai/clip-vit-base-patch32')
image_processor = clip_processor.image_processor
processor = DragonflyProcessor(image_processor=image_processor, tokenizer=tokenizer, image_encoding_style=model_type)
model = DragonflyForCausalLM.from_pretrained(model_name)
model = model.to(torch.bfloat16)
model = model.to(f"cuda:{device}")

# load the image
image = Image.open(image_path)
image = image.convert('RGB')
images = [image]
# images = None # if you do not want to pass any images

# process the text and image
text_prompt = format_text(question)
inputs = processor(text=[text_prompt], images=[image], max_length=1024, return_tensors="pt")
inputs['input_ids'] = inputs['input_ids'][0][:-1].unsqueeze(0)
inputs['attention_mask'] = inputs['attention_mask'][0][:-1].unsqueeze(0)
if "image_patches_indices" in inputs:
    inputs['image_patches_indices'] = inputs['image_patches_indices'][0][:-1].unsqueeze(0)
inputs = inputs.to(f"cuda:{device}")

# generate the response
with torch.inference_mode():
    output_ids = model.generate(
        **inputs,
        do_sample=True if temperature > 0 else False,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        eos_token_id=tokenizer.encode('<|eot_id|>'),
    )

outputs = processor.batch_decode(output_ids, skip_special_tokens=True)

# extract the response
response = []
for gen_text in outputs:
    gen_text_new = gen_text.split("assistant")[-1].strip(" ").strip("\n")
    gen_text_new = gen_text_new.split("<|eot_id|>")[0]
    response.append(gen_text_new)

pred = response[0].strip()
print(pred)