# Modeling Energy Consumption in Deep Learning Architectures Using Power Laws

This repository contains the official code, data, and analysis release accompanying our ECAI 2025 paper:
> **Modeling Energy Consumption in Deep Learning Architectures Using Power Laws**  
> _Mansour MAYAKI_  and _Victor Charpenay_
> 
> _ECAI 2025_  
> [DOI or arXiv link]


---

## 📚 Table of Contents

- [Overview](#-overview)
- [Repository Structure](#-repository-structure)
- [Practical Use Case: Energy-Aware Model Selection](#-practical-use-case-energy-aware-model-selection)
- [Datasets](#-datasets)
- [Results](#-results)
- [License](#-license)
- [Citation](#-citation)
- [Contributions](#-contributions)

---
---

## 📄 Overview

This paper bridges this gap by conducting a detailed empirical investigation into the energy efficiency of three widely used model architectures: LSTM, GRU, and Transformers. These architectures are fundamentally different in their computational design. LSTMs and GRUs are recurrent neural networks (RNNs) optimized for processing sequential data through memory mechanisms, while Transformers employ self-attention mechanisms, enabling superior performance on tasks involving long-range dependencies. Despite these differences, all three architectures are widely used in applications ranging from NLP to time series forecasting, making them ideal candidates for a comparative energy consumption study.
Our study focuses on measuring the energy consumption of these architectures during training under varying model configurations, such as the number of layers, hidden dimensions, and attention heads (for Transformers). Beyond raw energy measurements, we also explore the relationship between energy usage and model performance metrics, such as hardware efficiency and the number of floating point operations (FLOPs). Our goal is to show that it is possible to derive a general law that makes energy consumption predictable, given a model architecture.
It includes:
- Preprocessing scripts for data
- Training and evaluation of [your models]
- Code to reproduce figures and tables from the paper
- A small, publicly available dataset on [topic]

---

## 🗂 Repository Structure

```bash
.
├── data/               # Raw and processed data
├── notebooks/: Reproducible Jupyter notebooks used in the paper
  - UseCase.ipynb: Example application of our method
  - HEF_modelling.ipynb: Hardware Efficiency Factor modeling
  - Energy_Modeling.ipynb: Full pipeline for FLOPs, duration, and energy estimation
├── src/: Clean Python modules for FLOPs calculation, HEF modeling, and energy estimation
├── paper/              # LaTeX source files of the paper
├── results/            # Output: figures, tables, metrics
├── requirements.txt    # Python dependencies
├── LICENSE             # License for code and/or data
└── README.md           # This file


---

## 📦 Installation

### Using `pip`

```bash
git clone https://github.com/MansMayaki/MLEnergyConsumption.git
cd yourproject
pip install -r requirements.txt
```

## 🔋 Practical Use Case: Energy-Aware Model Selection

This section presents a practical use of our energy estimation method to support model selection based on energy efficiency.  
We compare two Transformer configurations on a GPU (**NVIDIA A100**, max throughput **156 TFLOPs/s**).  

Rather than benchmarking, our method uses architectural parameters to estimate energy consumption.

### Hardware and Input Settings
All steps are encapsulated in modular functions—define your model configuration and workload parameters (batch size, sequence length, layers, heads, embedding dimensionality $d_{\text{model}}$), and the system automatically computes the estimated energy consumption for one training epoch.

- **Max Throughput:** $v_{\text{max}} = 156 \times 10^{12}$ FLOPs/s  
- **Batch Size:** 64  
- **Sequence Length:** 320  

### Model Configurations
- **Model A:** 6 layers, $d_{\text{model}} = 512$, 8 attention heads  
- **Model B:** 12 layers, $d_{\text{model}} = 768$, 12 attention heads  

---
### **Step 1 & 2 — FLOPs, Hardware Efficiency & Duration Estimation**

FLOPs (in teraflops), efficiency values ηΘ(c), and operation durations tΘ(c) (in microseconds) for each model and operation.

| Operation           | Model | FLOPs (TF) | ηΘ(c) | tΘ(c) (µs) |
|---------------------|-------|------------|-------|------------|
| QKV Projections     | A     | 0.0322     | 35.31 | 35.08      |
| Attention Scores    | A     | 0.00674    | 7.75  | 33.28      |
| Attention Output    | A     | 0.00674    | 9.76  | 26.43      |
| Final Projection    | A     | 0.01074    | 12.19 | 33.87      |
| QKV Projections     | B     | 0.0724     | 51.19 | 108.90     |
| Attention Scores    | B     | 0.0100     | 10.43 | 74.21      |
| Attention Output    | B     | 0.0100     | 13.11 | 59.05      |
| Final Projection    | B     | 0.0240     | 21.04 | 88.31      |

---

### **Step 3 — Energy Estimation**

Regression model:

$$
E \approx 3.6318 - 0.1377 \cdot t_{\text{qkvProj}} + 0.5637 \cdot t_{\text{Score}} + 0.3041 \cdot t_{\text{FinalProj}} + 0.3041 \cdot t_{\text{AttnOut}}
$$

For Model A:

$$
E_A \approx  3.6318 - 0.1377 \cdot 35.08 + 0.5637 \cdot 33.28 + 0.3041 \cdot 33.87 + 0.3041 \cdot 26.43  = \mathbf{36\ J}
$$

For Model B:

$$
E_B \approx  3.6318 - 0.1377 \cdot 108.9 + 0.5637 \cdot 74.21 + 0.3041 \cdot 59.05 + 0.3041 \cdot 88.31 = \mathbf{ 79 \ J}
$$

---

### **Conclusion**
Despite being evaluated under identical input conditions, Model B is estimated to consume more than **twice energy per epoch** than Model A. While a single-epoch difference of roughly 42 joules may seem modest, the effect becomes pronounced at scale: over one million training epochs, this difference would accumulate to more than **11 kWh**. Such a gap is non-negligible in battery-powered, embedded, or high-throughput environments, where energy efficiency directly constrains deployment feasibility. 
The higher cost of Model B is largely attributable to its **increased dimensionality and number of layers**, which disproportionately amplify the computational burden of projection and score computations relative to Model A. Beyond highlighting the energy trade-offs between model sizes, this example illustrates how our method enables **early-stage architectural comparisons** and **energy-aware design choices** without requiring execution on actual hardware. 


---

## ▶️ How to Run

### 1. Usage

```bash
jupyter notebook notebooks/Energy_Modeling.ipynb
```

### 2. Reproduce Results

All figures and tables can be regenerated using the notebooks in `notebooks/`.

---

## 📊 Datasets

* 📂 `data/raw/`: Original measurements
* 📂 `data/processed/`: Cleaned datasets for modeling

> ⚠️ Due to size or license restrictions, some datasets may not be included directly. Please follow the instructions in `data/README.md` to download them.

---

## 📈 Results

Key results can be found in the `results/` folder, including:

* Plots and metrics
* LaTeX-formatted tables for publication

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
Please cite our work if you use any part of this project.

---

## 📚 Citation

```bibtex
@article{your2025paper,
  title={Your Paper Title},
  author={Author, A. and Researcher, B.},
  journal={Journal of Awesome ML Research},
  year={2025}
}
```

---

## 🤝 Contributions

Pull requests and issues are welcome. Please feel free to fork and adapt this work for your own research!

```



