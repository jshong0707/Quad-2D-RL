# 📦 Project Dependencies

This project was developed in a **Conda (Python 3.11)** environment for Reinforcement Learning and simulation with MuJoCo. Below are the key dependencies.

## Python
- Python: **3.11**

## Core Libraries
- [PyTorch](https://pytorch.org/): **2.4.1** (CUDA 12.1 build, NVIDIA RTX 3090 environment)  
  → For CPU-only, use `torch==2.4.1+cpu`
- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3): **2.6.0**
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium): latest (as of Sept 2025)

## Simulation
- [MuJoCo](https://github.com/google-deepmind/mujoco): **3.3.2**
- [glfw (Python binding)](https://pypi.org/project/glfw/): **2.6.2**
- [PyOpenGL](https://pypi.org/project/PyOpenGL/): **3.1.7**

## Utilities
- [NumPy](https://numpy.org/): **1.26.4**
- [SciPy](https://scipy.org/): latest (pip install)
- [Matplotlib](https://matplotlib.org/): latest
- [packaging](https://pypi.org/project/packaging/): latest

## System Requirements (Ubuntu)
- `libglfw3`, `libglfw3-dev` (GLFW3 shared library)
- `libgl1`, `libglu1-mesa`, `libx11-6`, `libxrandr2`, `libxinerama1`, `libxcursor1`, `libxi6`
- `mesa-utils` (for `glxinfo`)
- `ffmpeg` (optional, video export)

---

# 🚀 Installation (Conda)

```bash
# Create environment
conda create -n RL python=3.11 -y
conda activate RL

# PyTorch (GPU, CUDA 12.1 build)
pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# Core RL libraries
pip install stable-baselines3==2.6.0 gymnasium mujoco==3.3.2

# Rendering and plotting
pip install glfw==2.6.2 PyOpenGL==3.1.7 matplotlib scipy packaging
