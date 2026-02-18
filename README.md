# Flow Matching

A high-performance, pure PyTorch implementation of **Conditional Flow Matching (CFM)** with some improvements.


## 📁 Project Structure

```text
flowmatching_final/
├── src/
│   └── flowmatching/       # Core package
│       ├── config.py       # Configuration & Hyperparameters
│       ├── models.py       # Velocity Network (AdaLN)
│       ├── engine.py       # CFM Training Logic & EMA
│       ├── samplers.py     # ODE Solvers & Sampling
│       ├── data.py         # 2D Toy Datasets
│       └── utils.py        # Math: Solvers, Interpolants, OT
├── scripts/
│   ├── train.py            # Training script
│   └── demo.py             # End-to-end demo
├── setup.py                # Installation script
└── README.md
```

## ✨ Key Features

- **Advanced Solvers**: Explicit Midpoint (2nd order), RK4 (4th order), Dopri5 (Adaptive).
- **Optimal Transport (OT)**: Mini-batch OT coupling for straighter paths.
- **Trigonometric Paths**: Variance-preserving interpolants (`cos/sin` paths).
- **Architecture**: Adaptive Layer Norm (AdaLN) for time conditioning.
- **Training**: Logit-Normal time sampling focus on hard regions + EMA.

## 🚀 Installation

```bash
cd flowmatching_final
pip install -e .
```

## 🎮 Usage

You can run scripts directly without installation (via `sys.path` hack) or after installing.

### Quick Demo
Run training and sampling on the `8gaussians` dataset:
```bash
python scripts/demo.py --dataset 8gaussians --epochs 100 --ode-method midpoint
```

### Full Benchmark
Generate results for all datasets:
```bash
python scripts/demo.py --all-datasets
```
