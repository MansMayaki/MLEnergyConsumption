# Modeling Energy Consumption in Deep Learning Architectures Using Power Laws

This repository contains the code, data, and analysis accompanying the research paper:

> **Modeling Energy Consumption in Deep Learning Architectures Using Power Laws**  
> _Mansour MAYAKI_  
> _ECAI 2025_  
> [DOI or arXiv link]

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
├── notebooks/          # Jupyter notebooks with analysis and experiments
├── src/                # Python scripts for preprocessing, training, etc.
├── paper/              # LaTeX source files of the paper
├── results/            # Output: figures, tables, metrics
├── requirements.txt    # Python dependencies
├── environment.yml     # (optional) Conda environment
├── LICENSE             # License for code and/or data
└── README.md           # This file


---

## 📦 Installation

### Option 1: Using `pip`

```bash
git clone https://gitlab.com/yourusername/yourproject.git
cd yourproject
pip install -r requirements.txt
```

### Option 2: Using Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate your-env-name
```

---

## ▶️ How to Run

### 1. Preprocess Data

```bash
python src/preprocess.py
```

### 2. Train Model

```bash
python src/train_model.py
```

### 3. Reproduce Results

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



