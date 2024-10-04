# Gamma-MOD: Mixture-of-Depth Adaptation for Multimodal Large Language Models

![Gamma-MOD Banner](link_to_banner_image)

[![Version](https://img.shields.io/badge/version-1.0.0-blue)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](link_to_license)
[![Contact](https://img.shields.io/badge/Contact-Yaxin%20Luo-green)](mailto:yaxin.luo@example.com)

**Gamma-MOD** is a novel approach to enhance computational efficiency in Multimodal Large Language Models (MLLMs) by incorporating **Mixture-of-Depth (MoD)** layers. This plug-and-play strategy seamlessly replaces redundant dense layers, significantly reducing computational costs while maintaining performance.

---

## 🔗 Table of Contents
- [Overview](#overview)
- [Motivation](#motivation)
- [Key Contributions](#key-contributions)
- [Model Architecture](#model-architecture)
- [Efficiency Gains](#efficiency-gains)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Running the Model](#running-the-model)
- [Experiments](#experiments)
- [Results](#results)
- [Citation](#citation)
- [Contact](#contact)
- [License](#license)

---

## 🚀 Overview

![Logo or Key Visual](link_to_logo_or_key_image)

**Gamma-MOD** is inspired by **activated tokens** and aims to identify layer-level redundancy using a novel metric called **ARank** (Rank of Attention Maps). By transforming dense MLLM layers into **Mixture-of-Depth** layers, Gamma-MOD saves on computational costs while preserving model accuracy.

### Key Features:
- **ARank Metric**: Guides the replacement of redundant layers with MoD layers.
- **Shared Vision-Language Router**: Facilitates cross-modality token routing.
- **Masked Routing Learning**: Prevents critical tokens from being skipped during model adaptation.

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

![Model Diagram](link_to_model_architecture_image)

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

| ![Training Efficiency](link_to_training_efficiency_image) | ![Inference Gains](link_to_inference_gains_image) |
|:--:|:--:|
| Training Efficiency | Inference Gains |

---

## 🛠️ Getting Started

### Installation

First, clone the repository:

```bash
git clone https://github.com/Yaxin9Luo/Gamma-MOD.git
```

### Prerequisites

- Python >= 3.8
- PyTorch >= 1.12
- Additional dependencies can be found in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Running the Model

To run Gamma-MOD with a pre-trained MLLM:

```bash
python run_gamma_mod.py --config configs/model_config.yaml
```

---

## 🔬 Experiments

Gamma-MOD was tested on **three popular MLLMs** across **9 benchmark datasets**.

- **LLaVA-HR**: Training time reduced by **31%** and inference time by **53.2%**, with only **1.5%** accuracy drop.
- **Generalization**: Demonstrated the ability to generalize across different MLLMs.

![Experimental Results](link_to_experimental_results_image)

---

## 📈 Results

| Model       | Training Time Reduction | Inference Time Reduction | Accuracy Drop |
|-------------|-------------------------|--------------------------|---------------|
| LLaVA-HR    | 31.0%                   | 53.2%                    | 1.5%          |
| Mini-Gemini | 25.6%                   | 48.1%                    | 1.8%          |

For more details, check the [full report](link_to_experiment_report).

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

---

![Gamma-MOD Footer Banner](link_to_footer_banner_image)