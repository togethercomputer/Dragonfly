# Dragonfly

*Multi-Resolution Zoom Supercharges Large Visual-Language Model*

## 🔥 News
- [Our paper](todo) is out on arxiv.
- Check out [uur blogpost](todo).


## 📖 Introduction

![Dragonfly framework](assets/model_overview.pdf)

Recent advances in large multimodal models (LMMs) suggest that higher image resolution enhances the fine-grained understanding of image details, crucial for tasks such as visual commonsense reasoning and analyzing biomedical images. However, increasing input resolution poses two main challenges: 1) It extends the context length required by the language model, leading to inefficiencies and hitting the model's context limit; 2) It increases the complexity of visual features, necessitating more training data or more complex architecture. We introduce Dragonfly, a new LMM architecture that enhances fine-grained visual understanding and reasoning about image regions to address these challenges. Dragonfly employs two key strategies: multi-resolution visual encoding and zoom-in patch selection. These strategies allow the model to process high-resolution images efficiently while maintaining reasonable context length. Our experiments on eight popular benchmarks demonstrate that Dragonfly achieves competitive or better performance compared to other architectures, highlighting the effectiveness of our design. Additionally, we finetuned Dragonfly on biomedical instructions, achieving state-of-the-art results on multiple biomedical tasks requiring fine-grained visual understanding, including 92.3% accuracy on the Path-VQA dataset (compared to 83.3% for Med-Gemini) and the highest reported results on biomedical image captioning. To support model training, we curated a visual instruction-tuning dataset with 5.5 million image-instruction samples in the general domain and 1.4 million samples in the biomedical domain. We also conducted ablation studies to characterize the impact of various architectural designs and image resolutions, providing insights for future research on visual instruction alignment.


# 📖 Table of Contents
1. [Installation](#installation)
2. [Checkpoint](#checkpoint)
3. [Dataset](#dataset)
4. [Training](#training)
5. [Inference](#inference)
6. [BibTeX](#bibtex)
7. [Licence](#license)


<a name="installation"/>

# 💿 Installation

Please use the following steps to create an environment for running Dragonfly.

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

Install additional packages for training cases
```bash
pip install packaging
pip uninstall -y ninja && pip install ninja
pip install flash-attn --no-build-isolation
```

<a name="training"/>

# Training
training data format:
{"image_url": , "conversations":, "source":,} (We will release the data soon).

<a name="inference"/>

# Inference
How to perform the one pass inference. Refer test_dragonfly.py

<a name="checkpoint"/>

# Checkpoint

Dragonfly-v1-llama8b
Dragonfly-med-v1-llama8b
license: research-only

<a name="dataset"/>

# Dataset

I will release it soon on HF hub. 

<a name="bibtex"/>

# BibTeX

```bibtex
TODO
```
<a name="license"/>

# 🪪 License

MIT License
