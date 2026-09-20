# Self-Training Antichess AI Engine

An **AlphaZero-inspired Deep Reinforcement Learning engine** for **Antichess** (Losing Chess), combining a **4-block Residual Neural Network (ResNet)** with **Monte Carlo Tree Search (MCTS)** in PyTorch [4, 8].

---

## 📌 Project Overview

This project implements an autonomous chess variant engine that learns strategy from scratch (*tabula rasa*) purely through unsupervised self-play [4, 17]. Antichess is a solved chess variant where the objective is to lose all pieces or become stalemated, and captures are strictly forced [7, 19]. 

By pairing a dual-headed deep convolutional network (Policy & Value heads) with MCTS, the engine replaces handcrafted evaluation heuristics with deep neural intuition, outperforming unguided MCTS search algorithms [4, 9, 66].

---

## ✨ Key Features

* **AlphaZero Architecture**: Shared deep ResNet backbone branching into a Policy head (move probability distribution) and Value head (board position evaluation) [4, 9, 38].
* **ResNet-4 Backbone**: Engineered with 4 residual blocks and skip connections to eliminate vanishing gradients while maintaining fast execution speed (~15 min/generation vs ~56 min for DenseNet) [4, 28, 53].
* **Monte Carlo Tree Search (MCTS)**: Evaluates promising tactical variations using neural network predictions to balance exploration and exploitation [4, 21].
* **Optimized Pipeline & Data Unpacking**: Fixed multidimensional batching errors via tuple data unpacking (`for b, p, r in batch`), boosting dataset compilation speed by **2.5x**.
* **Stochastic Gradient Descent (SGD)**: Trained using SGD with momentum, avoiding convergence failures observed with Adam in self-play loss landscapes [11, 50].
* **Interactive Desktop GUI**: Built using **Pygame** and **python-chess** (800x800 resolution) featuring forced capture rule validation, piece selection highlighting, and zero-temperature evaluation mode [46, 47].

---

## 🏗️ System Architecture

```
                      +-------------------+
                      |   GameState       |
                      | (python-chess)    |
                      +---------+---------+
                                |
                                v
                      +-------------------+
                      | Monte Carlo Tree  |
                      |   Search (MCTS)   |
                      +---------+---------+
                                |
                                v
                      +-------------------+
                      | ResNet-4 Model    |
                      | (PyTorch)         |
                      +----+---------+----+
                           |         |
             +-------------+         +-------------+
             |                                     |
             v                                     v
    +-----------------+                   +-----------------+
    |   Policy Head   |                   |   Value Head    |
    | (4,216 actions) |                   |  (-1.0 to +1.0) |
    +-----------------+                   +-----------------+
```

### Neural Network Specifications
* **Input Board Representation**: $12 \times 8 \times 8$ tensor encoding piece types and colors [4, 9].
* **Shared Core**: 2D Convolution $\rightarrow$ Batch Normalization $\rightarrow$ ReLU $\rightarrow$ 4 Residual Blocks [34, 35].
* **Policy Output**: Vector of length 4,216 representing probability distribution across all legal move combinations [4, 39].
* **Value Output**: Scalar evaluation predicting win probability ($v \in [-1, 1]$) [4, 38].

---

## 🛠️ Repository Structure

```
├── _1_engine.py         # Game state manager & forced-capture rule engine
├── _2_model.py          # PyTorch ResNet-4 dual-head neural network
├── _5_mcts.py           # Monte Carlo Tree Search implementation (.pop() move selector)
├── _7_gui.py            # Pygame desktop client (800x800) with asset loader
├── train.py             # PyTorch training loop with tuple data unpacking
├── self_play.py         # Unsupervised generation loop (200 games / gen)
├── chess_pieces/        # Graphical assets (WP.png, BB.png, etc.)
└── antichess_model.pth  # Trained model weights checkpoint
```

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.10+ installed along with the required dependencies:

```bash
pip install torch pygame python-chess numpy
```

### 1. Model Training (Self-Play Generation)

To execute self-play game generation and train the PyTorch neural network:

```bash
python self_play.py
```

> **Training Metrics**: Each generation plays **200 self-play games**, generating approximately **14,800 unique board positions** for batch training [52].

### 2. Playing Against the AI (GUI)

To play locally against the trained model in the interactive Pygame window:

```bash
python _7_gui.py
```

* **Controls**: 
  * **Click 1**: Select a piece (highlighted in light blue).
  * **Click 2**: Choose destination square.
* **Temperature Setting**: The GUI executes MCTS at `temperature=0.0` for competitive play, forcing the AI to exploit its best evaluated move.

---

## 📊 Technical Highlights & Benchmark Insights

| Parameter | Selected Approach | Alternative Evaluated | Key Rationale |
|---|---|---|---|
| **Architecture** | **ResNet (4 Blocks)** | DenseNet-4 / ResNet-16 | Equal validation accuracy; ResNet-4 was **3.7x faster** per generation [53]. |
| **Optimizer** | **SGD + Momentum** | Adam | Adam failed to converge; SGD achieved steady loss minimization [11, 50]. |
| **Batch Unpacking** | **Tuple Unpacking** | Direct Array Stacking | Solved inhomogeneous tuple shape errors during tensor conversion. |
| **Move Selection** | **List `.pop()` Method** | Direct indexing | Bypassed list wrapper bugs in `random.choices` for `python-chess` Move extraction. |

---

## 📜 References & Acknowledgements

* **AlphaZero Foundation**: Silver et al., *Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm* (arXiv:1712.01815) [8].
* **Antichess DRL Study**: Chalmers University of Technology & University of Gothenburg Bachelor's Thesis (2024) [1, 2].
* **Libraries**: Built using [python-chess](https://python-chess.readthedocs.io/), [PyTorch](https://pytorch.org/), and [Pygame](https://www.pygame.org/).
