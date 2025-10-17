# Composite-CBF
This repository implements codes to learn a neural network control barrier function (CBF) that satisfies complex safety constraints, e.g., constraints composited by logical operations. Actuation limits are considered and the CBF is trained to approximate the maximal control invariant set inside the safe region. The details of this learning method are reported in the following paper:

[Learning Performance-oriented Control Barrier
Functions Under Complex Safety Constraints and
Limited Actuation](https://arxiv.org/abs/2401.05629) \
Lakshmideepakreddy Manda, Shaoru Chen, Mahyar Fazlyab \
Conference on Robot Learning (CoRL), 2024

## Introduction
For a control-affine system $\dot{x} = f(x) + g(x) u$, a CBF gives rise to a numerically efficient online safety filter, CBF-QP, that has found wide applications in robotic systems. However, the offline synthesis of CBF has been a long-standing challenge. `Composite-CBF` attempts to fully automate the CBF design by learning a neural network CBF that addresses the following practical concerns:

1. **Complex Safety Constraints**: The safe set for a system can be given by Boolean logical operation on multiple constraints.
2. **High Relative Degree**: The safety constraints are often defined on a subset of states that are not directly actuated and cannot be used as CBF
3. **Bounded Actuation**: There is always actuation limit for every system. Taking bounded actuation into account significantly complicates CBF design. 
4. **Volume Maximization**: We want to find a CBF that approximates the maximal control invariant set contained in the safe region. 

`Composite-CBF` takes all above challenges into account through (i) a novel NN CBF parameterization and (ii) a two-phase learning algorithm based on [Hamilton-Jacobi reachability analysis](https://arxiv.org/abs/1709.07523). 

### Example: Double Integrator with Splitted Safety Regions

For a double integrator system, the safe region in the state space of (position, velocity) or $(p, v)$ is separated into two disconnected parts. `Composite-CBF` is able to learn a NN CBF $h(x)$ whose $0$-superlevel set (green curve) approximates the maximal control invariant set in each separate safe regions. The feasibility of the CBF condition is evaluated at the sampled states with infeasible points labeled as white squares. At states where the CBF-QP is feasible, the Chebyshev radius of the safe control set $\lbrace u \mid -1 \leq u \leq 1, \frac{\partial h}{\partial x} (f(x) + g(x) u ) + \gamma h(x) \geq 0 \rbrace$ (with minimum value $0$ and maximum value $1$ in this example) is denoted by the heatmap. 

<img src="https://github.com/ShaoruChen/web-materials/blob/main/L4DC_composite_CBF/final_result.png" width=400, height=360> 


## Installation
The following conda environment can be created to run the codes. 

```
conda create --name composite_cbf python=3.10
conda activate composite_cbf
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 -c pytorch
pip install -r requirements.txt
```

Run the following codes to train NN CBFs and recover the results reported in the main paper:

```
python examples/DI_box/main_DI_box.py --train --num_epochs 300 --checkpoint_freq 30 --objective cbvf --seed 0 --plot_level_set
python examples/DI_obstacle/main_DI_obstacle.py --train --num_epochs 300 --checkpoint_freq 30 --objective cbvf --seed 0 --plot_level_set
python examples/DI_split/main_DI_split.py --train --num_epochs 300 --checkpoint_freq 30 --objective cbvf --seed 0 --plot_level_set 
```
