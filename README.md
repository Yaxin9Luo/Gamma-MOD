# Gamma-MOD: Mixture-of-Depth Adaptation for Multimodal Large Language Models

![Gamma-MOD Banner](/asset/gamma_logo.jpg)

[![Version](https://img.shields.io/badge/version-1.0.0-blue)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](link_to_license)
[![Contact](https://img.shields.io/badge/Contact-Yaxin%20Luo-green)](mailto:yaxin.luo@example.com)

**Gamma-MOD** is a novel approach to enhance computational efficiency in Multimodal Large Language Models (MLLMs) by incorporating **Mixture-of-Depth (MoD)** layers. This plug-and-play strategy seamlessly replaces redundant dense layers, significantly reducing computational costs while maintaining performance.

---

## 🔗 Table of Contents
- [Overview](#-overview)
- [Visualization Results](#-visualization-results)
- [Motivation](#-motivation)
- [Key Contributions](#-key-contributions)
- [Model Architecture](#-model-architecture)
- [Efficiency Gains](#-efficiency-gains)
- [Getting Started](#-getting-started)
  - [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Running the Model](#running-the-model)
- [Experiments](#-experiments)
- [Results](#-results)
- [Citation](#-citation)
- [Contact](#-contact)
- [License](#-license)

---

## 🚀 Overview

**Gamma-MOD** is inspired by **activated tokens** and aims to identify layer-level redundancy using a novel metric called **ARank** (Rank of Attention Maps). By transforming dense MLLM layers into **Mixture-of-Depth** layers, Gamma-MOD saves on computational costs while preserving model accuracy.

### Key Features:
- **ARank Metric**: Guides the replacement of redundant layers with MoD layers.
- **Shared Vision-Language Router**: Facilitates cross-modality token routing.
- **Masked Routing Learning**: Prevents critical tokens from being skipped during model adaptation.
![Gamma-MOD Banner](/asset/motivations.png)

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
## 🎯 Motivation

Despite advancements in MLLMs, their **high computational costs** limit practical application, particularly for real-time inference. While **Mixture-of-Experts (MoE)** techniques help reduce active parameters, Gamma-MOD reduces **activated tokens**, offering superior efficiency.

---

## ✨ Key Contributions

1. **Mixture-of-Depth Framework**: Introduces a computationally efficient framework to transform dense MLLM layers into sparse MoD layers.
2. **ARank Metric**: Provides a method to identify layer-level redundancy for strategic deployment of MoD layers.
3. **Experimental Validation**: Achieves **51.6% reduction in computational costs** with minimal performance drop (~1.5%).

---

## 🏗️ Model Architecture

Gamma-MOD involves two main stages:

1. **Vision-Language Alignment**: Aligns visual and textual modalities, using ARank to identify redundant layers.
2. **Instruction Tuning**: Integrates MoD layers, with **masked routing learning** to avoid skipping critical tokens.

![Model Diagram](/asset/model_arch.png)

### Example Flow:
1. **Input Processing**: Visual and text features are aligned.
2. **Routing with ARank**: Redundant dense layers are replaced.
3. **Final Tuning**: Optimizes MoD layers for minimal computational costs.

---

## 📊 Efficiency Gains

Gamma-MOD results in significant efficiency improvements:
- **Training time**: Reduced by **31%**.
- **Inference time**: Reduced by **53.2%**.
- **FLOPs Reduction**: **51.6%** with minimal impact on accuracy.

| Training Efficiency |
![Training Efficiency](/asset/Efficiency.png)


---

## 🛠️ Getting Started

### Installation

1. Clone the repository and navigate to the LLaVA-HR folder:

```bash
git clone https://github.com/luogen1996/LLaVA-HR.git
cd LLaVA-HR
```

2. Create and activate a new conda environment:

```bash
conda create -n llava-hr python=3.10 -y
conda activate llava-hr
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

For questions, please reach out to [Yaxin Luo](mailto:yaxin.luo@example.com).

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](link_to_license) file for details.

---

## 👀 Acknowledgments

Special thanks to all contributors and the LLaVA project for inspiration.
