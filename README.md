<div align="center">
  <img src="assets/dragonfly_icon.png" alt="Dragonfly" style="width: 150px; display: block; margin-left: auto; margin-right: auto;" />
  <h1>Dragonfly: Multi-Resolution Zoom Supercharges Large Visual-Language Model</h1>
</div>

## 🔥 News
- [Our paper](todo) is out on arxiv.
- Check out [our blogpost](todo).
- Our model checkpoints are out on huggingface 🤗 🚀: 
    - General: `togethercomputer/Dragonfly-v1-llama8b` 
    - Biomed: `togethercomputer/Dragonfly-med-v1-llama8b`


## 📖 Introduction

![Dragonfly framework](assets/model_overview.png)

Recent advances in large multimodal models (LMMs) suggest that higher image resolution enhances the fine-grained understanding of image details, crucial for tasks such as visual commonsense reasoning and analyzing biomedical images. However, increasing input resolution poses two main challenges: 1) It extends the context length required by the language model, leading to inefficiencies and hitting the model's context limit; 2) It increases the complexity of visual features, necessitating more training data or more complex architecture. We introduce Dragonfly, a new LMM architecture that enhances fine-grained visual understanding and reasoning about image regions to address these challenges. Dragonfly employs two key strategies: multi-resolution visual encoding and zoom-in patch selection. These strategies allow the model to process high-resolution images efficiently while maintaining reasonable context length. Our experiments on eight popular benchmarks demonstrate that Dragonfly achieves competitive or better performance compared to other architectures, highlighting the effectiveness of our design. Additionally, we finetuned Dragonfly on biomedical instructions, achieving state-of-the-art results on multiple biomedical tasks requiring fine-grained visual understanding, including 92.3% accuracy on the Path-VQA dataset (compared to 83.3% for Med-Gemini) and the highest reported results on biomedical image captioning. To support model training, we curated a visual instruction-tuning dataset with 5.5 million image-instruction samples in the general domain and 1.4 million samples in the biomedical domain. We also conducted ablation studies to characterize the impact of various architectural designs and image resolutions, providing insights for future research on visual instruction alignment.


# 📖 Table of Contents
1. [Installation](#installation)
2. [Checkpoint](#checkpoint)
5. [Inference](#inference)
3. [Dataset](#dataset)
4. [Training](#training)
6. [BibTeX](#bibtex)
7. [Licence](#license)


<a name="installation"/>

## 💿 Installation

Clone this repository and navigate to LLaVA folder
```bash
git clone https://github.com/togethercomputer/Dragonfly.git
cd Dragonfly
```

Create a conda environment and install necessary packages
```bash
conda env create -f environment.yml
conda activate dragonfly_env
```

Install flash attention
```bash
pip install flash-attn --no-build-isolation
```

As a final step, please run the following command. 
```bash
pip install --upgrade -e .
```

<a name="checkpoint"/>

## 🏁 Checkpoint

*Note: These models are released under Creative Commons Attribution Non Commercial 4.0*

We release two huggingface model checkpoints: `togethercomputer/Dragonfly-med-v1-llama8b` and `togethercomputer/Dragonfly-v1-llama8b`. Please follow the script `pipeline/train/testing.py` for more details. We provide a brief description on how to use them below.

<a name="inference"/>

## 🧠 Inference

If you have successfully completed the [Installation](#installation) process, then you should be able to follow the steps below. 

We provide two test examples inside `pipeline/train/test_images`. 

Load necessary packages
```python
import sys
from PIL import Image
import torch
from dragonfly.models.modeling_dragonfly import *
from dragonfly.models.processing_dragonfly import *
from transformers import AutoProcessor, AutoTokenizer
```

Instantiate the tokenizer, processor, and model. 
```python
tokenizer = AutoTokenizer.from_pretrained("togethercomputer/Dragonfly-med-v1-llama8b")
clip_processor = AutoProcessor.from_pretrained('openai/clip-vit-base-patch32')
image_processor = clip_processor.image_processor
processor = DragonflyProcessor(image_processor=image_processor, tokenizer=tokenizer, image_encoding_style="llava-hd")
model = DragonflyForCausalLM.from_pretrained("togethercomputer/Dragonfly-med-v1-llama8b")
model = model.to(torch.bfloat16)
model = model.to(f"cuda:0")
```

Now, lets load the image and process them.
```python
image = Image.open("./test_images/chext-xray.jpeg")
image = image.convert('RGB')
images = [image]
# images = None # if you do not want to pass any images

question = "are the lungs normal appearing?"
text_prompt = format_text(question)
inputs = processor(text=[text_prompt], images=[image], max_length=1024, return_tensors="pt")
inputs['input_ids'] = inputs['input_ids'][0][:-1].unsqueeze(0)
inputs['attention_mask'] = inputs['attention_mask'][0][:-1].unsqueeze(0)
if "image_patches_indices" in inputs:
    inputs['image_patches_indices'] = inputs['image_patches_indices'][0][:-1].unsqueeze(0)
inputs = inputs.to(f"cuda:0")
```

Finally, let us generate the responses from the model
```python
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
```

<a name="dataset"/>

## 📊 Dataset

I will release it soon on HF hub. 

<a name="training"/>

## 🏋️‍♂️ Training

*Note: This training recipe is specifically for our general domain model.*

We adopt a two-stage training process.

### Stage 1
In this stage, we only train our projection layer, so that the model learns to map the embeddings from the vision encoder into the LLM space. The dataset mixture used in this stage is `stage1_dataset`, which contains short image and caption pairs.

```bash
sh train_dragonfly_stage1.sh
```

### Stage 2
In this stage, we train our vision encoder, projection layer, and LLM jointly on image and text data. Our training dataset mixture for this stage is provided in `stage2_dataset`. This dataset contains xx.xx% of text-only dataset as well. We also include a math dataset, given in `math_instruct`. 

```bash
sh train_dragonfly_stage2.sh
```

Please ensure to update the paths inside the bash script according to your local file paths.

For both stages, the dataset is formatted in the following manner:

```json
{
  "image_url": "<path_to_image>",
  "conversations": "<text_data_formatted>",
  "source": "<data_source>"
}
```

Conversation format follows standard Llama3 as follows. 

```plaintext
<|start_header_id|>user<|end_header_id|>

Describe the content in the image.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

<a name="bibtex"/>

## 📚 BibTeX

```bibtex
TODO
```
<a name="license"/>

## 🪪 License

[CC BY-NC 4.0](LICENSE)
