# Gamma-MOD: Mixture-of-Depth Adaptation for Multimodal Large Language Models
<p align="center">
     <img src="./asset/gamma_logo_processed.jpg" alt="Gamma-MOD Banner" width="50%">
</p>

[![Version](https://img.shields.io/badge/version-1.0.0-blue)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](link_to_license)
[![Contact](https://img.shields.io/badge/Contact-Yaxin%20Luo-green)](mailto:yaxin.luo@example.com)

**Gamma-MOD** is a novel approach to enhance computational efficiency in Multimodal Large Language Models (MLLMs) by incorporating **Mixture-of-Depth (MoD)** layers. This plug-and-play strategy seamlessly replaces redundant dense layers, significantly reducing computational costs while maintaining performance.

---

## 🔗 Table of Contents
- [Overview](#-overview)
- [Visualization Results](#-visualization-results)
- [Getting Started](#-getting-started)
  - [Installation](#installation)
  - [Data Preparation](#data-preparation)
  - [Training](#training)
- [Experiments](#-experiments)
- [Results](#-results)
- [Citation](#-citation)
- [Contact](#-contact)
- [License](#-license)

---

## 🚀 Overview

**Gamma-MOD** is a novel approach to enhance computational efficiency in Multimodal Large Language Models (MLLMs) by incorporating **Mixture-of-Depth (MoD)** layers. This plug-and-play strategy seamlessly replaces redundant dense layers, significantly reducing computational costs while maintaining performance.

### Motivation

Despite recent advancements in MLLMs, their high computational demands have limited practical applications, especially for real-time inference. Traditional Mixture-of-Experts (MoE) techniques have attempted to address this issue, but often fall short in achieving optimal efficiency. Gamma-MOD tackles this challenge by introducing a new paradigm that focuses on reducing activated tokens, offering superior efficiency compared to existing methods. Our approach is inspired by the concept of activated tokens and aims to transform dense MLLM layers into sparse MoD layers, ultimately making MLLMs more accessible and applicable in resource-constrained environments.

### Key Features:
- **ARank Metric**: Guides the replacement of redundant layers with MoD layers.
- **Shared Vision-Language Router**: Facilitates cross-modality token routing.
- **Masked Routing Learning**: Prevents critical tokens from being skipped during model adaptation.
![Gamma-MOD Banner](/asset/model_arch.png)

### 📊 Efficiency Gains

Gamma-MOD results in significant efficiency improvements:
- **Training time**: Reduced by **31%**.
- **Inference time**: Reduced by **53.2%**.
- **FLOPs Reduction**: **51.6%** with minimal impact on accuracy.
![Training Efficiency](/asset/Efficiency.png)
---
## 🎨 Visualization Results

Our Gamma-MOD approach demonstrates impressive efficiency in routing tokens and focusing on critical information. Fig. 4 illustrates these results visually.

### Key Observations:
![Visualization of Routing and Skipped Content](/asset/visualization.png)
1. **Consistent Routing Patterns** (Fig. 4a):
   - Question tokens are mostly retained
   - Image tokens show the highest redundancy and are routed the most
   - Response tokens fall between these two extremes

2. **Efficient Content Skipping** (Fig. 4b):
   - Gray areas in images represent skipped tokens (often background or less relevant pixels)
   - White areas highlight regions the model focuses on more intensely

3. **Improved Focus on Critical Information**:
   - By routing out redundant tokens, the model can allocate more computational resources to important areas
   - Example: In the IQ test image (middle of first row), the model concentrates on arithmetic and geometric aspects, leading to more accurate responses

This visualization demonstrates how Gamma-MOD effectively reduces computational overhead while maintaining the model's ability to process and respond to complex multimodal inputs.

---

## 🛠️ Getting Started

### Installation
(Notice: Install the required packages and versions for the model you wish to modify to MoD version, below is for LLaVA-HR, for Mini-Gemini, just upgrade transformers to 4.36.2 as the official version)
1. Clone the repository and navigate to the Gamma-MOD folder:

```bash
git clone https://github.com/Yaxin9Luo/Gamma-MOD.git
cd Gamma-MOD
```

2. Create and activate a new conda environment:

```bash
conda create -n gamma-mod python=3.10 -y
conda activate gamma-mod
```

3. Upgrade pip and install the package:

```bash
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

4. Install additional packages for training:

```bash
pip install ninja
pip install flash-attn --no-build-isolation
```
### Data Preparation
Please refer to the original [LLaVA-HR](https://github.com/luogen1996/LLaVA-HR/tree/main) and [Mini-Gemini](https://github.com/dvlab-research/MGM/tree/main) for data preparation. Or whatever MLLM's offical repo you are using.

### Training
#### Stage 1: Pretraining
Please download the caption annotations blip_laion_cc_sbu_558k.json and images from here. Move the downloaded files to the /data/data folder. Then run the following command to start the training process:
```bash
bash bash scripts/v1_5/pretrain_llava_hr.sh
```
We recommend to directly pre-trained projector, here are the link from official LLaVA-HR and Mini-Gemini.
| Version | Vision Encoder | Projection | Pretrain Data | Pretraining schedule | Download |
|---------|----------------|------------|---------------|----------------------|----------|
| LLaVA-HR-7b | CLIP-L & ConvNeXt-L | MLP-2x | LCS-558K | 1e | [projector](https://huggingface.co/favor123/llava-hr-7b-pretrain-384) |
| LLaVA-HR-X-13b | CLIP-L & ConvNeXt-XXL | MLP-2x | LCS-558K | 1e | [projector](https://huggingface.co/favor123/llava-hr-13b-x-sft-1024) |
| Mini-Gemini-HD-7b | CLIP-L | MLP-2x | MGM-Pretrain | 1e | [projector](https://huggingface.co/YanweiLi/MGM-Pretrain) |


#### Stage 2: Gamma-MOD Fine-Tuning
##### Step 1: ARank analysis
Please run the stage-1 alignment model on any dataset you wish to compute the ARank.We will use sqa as an example. 
```bash
bash scripts/v1_5/eval_full/arank.sh /path/to/your/stage1_checkpoint 
```
We also provide the stage-1 checkpoint for your convenience.
| Version | Download |
|---------|----------|
| Gamma-MoD-llava-hr-7b-stage1 | [model](https://huggingface.co/YaxinLuo/Gamma-MoD-llava-hr-7b-stage1) |
| Gamma-MoD-Mini-Gemini-HD-7b-stage1 | [model](https://huggingface.co/YaxinLuo/Gamma-MoD-Mini-Gemini-HD-7b-stage1) |
##### Step 2: Fine-Tuning
After you get the ARank, you can use the ARank to replace the dense layers in the original model. Reference to llava_llama_mod.py file and the initialize_mod_modules function.
Then train the model with the following command:
```bash
bash /path/to/your/fine_tune_mod.sh
```
We also provide the stage-2 sft checkpoint for your convenience.
| Version | Download |
|---------|----------|
| Gamma-MoD-llava-hr-7b | [model](https://huggingface.co/YaxinLuo/Gamma-MoD-llava-hr-7b) |
| Gamma-MoD-llava-hr-13b | [model](https://huggingface.co/YaxinLuo/Gamma-MoD-llava-hr-13b) |
| Gamma-MoD-Mini-Gemini-HD-7b | [model](https://huggingface.co/YaxinLuo/Gamma-MoD-Mini-Gemini-HD-7b) |
---
## 🔬 Experiments

Gamma-MOD was tested on **three popular MLLMs** across **9 benchmark datasets**.

- **LLaVA-HR**: Training time reduced by **31%** and inference time by **53.2%**, with only **1.5%** accuracy drop.
- **Generalization**: Demonstrated the ability to generalize across different MLLMs.

![Experimental Results](/asset/compare_others.png)
![Experimental Results](/asset/scalable.png)

---

## 📈 Results

| Model       | Training Time Reduction | Inference Time Reduction | Accuracy |
|-------------|-------------------------|--------------------------|---------------|
| $\gamma$-MoD-LLaVA-HR-7B    | 31.0%                   | 53.2%                    | -1.5%          |
| $\gamma$-MoD-LLaVA-HR-13B    | 18.8%                   | 50.4%                    | -0.3%         |
| $\gamma$-MoD-LLaVA-HR-X-13B    | 17.4%                 | 58.6%                    | +0.4%          |

For more details, check the [full report](link_to_experiment_report).(Coming soon!!!!!!!!!!)

---

## 📖 Citation

If you use Gamma-MOD in your work, please cite:

```bibtex
@inproceedings{anonymous2025gammaMOD,
  title={Gamma-MOD: Exploring Mixture-of-Depth Adaptation for Multimodal Large Language Models},
  author={Anonymous},
  booktitle={International Conference on Learning Representations},
  year={2025}
}
```

---

## 📧 Contact

For questions, please reach out to [Yaxin Luo](yaxinluo999@163.com).

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👀 Acknowledgments

Special thanks to all contributors and the LLaVA project for inspiration.
