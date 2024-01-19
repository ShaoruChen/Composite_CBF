# Composite-CBF
This repository implements codes to learn a neural network control barrier function (CBF) that satisfies complex safety constraints, e.g., constraints composited by logical operations. Actuation limits are considered and the CBF is trained to approximate the maximal control invariant set inside the safe region. The details of this learning method are reported in the following paper:

[Learning Performance-Oriented Control Barrier Functions Under Complex Safety Constraints and Limited Actuation](https://arxiv.org/abs/2401.05629) \
Shaoru Chen, Mahyar Fazlyab
(Submitted to L4DC 2024)

## Introduction
For a control-affine system $\dot{x} = f(x) + g(x) u$, a CBF gives rise to a numerically efficient online safety filter, CBF-QP, that has found wide applications in robotic systems. However, the offline synthesis of CBF has been a long-standing challenge. `Composite-CBF` attempts to fully automate the CBF design by learning a neural network CBF that addresses the following practical concerns:

1. **Complex Safety Constraints**: The safe set for a system can be given by Boolean logical operation on multiple constraints.
2. **High Relative Degree**: The safety constraints are often defined on a subset of states that are not directly actuated and cannot be used as CBF
3. **Bounded Actuation**: There is always actuation limit for every system. Taking bounded actuation into account significantly complicates CBF design. 
4. **Volume Maximization**: We want to find a CBF that approximates the maximal control invariant set contained in the safe region. 

`Composite-CBF` takes all above challenges into account through (i) a novel NN CBF parameterization and (ii) a two-phase learning algorithm based on [Hamilton-Jacobi reachability analysis](https://arxiv.org/abs/1709.07523). 

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
