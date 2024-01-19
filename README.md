# composite-CBF
This repository implements codes to learn a neural network control barrier function (CBF) that satisfies complex safety constraints, e.g., constraints composited by logical operations. Actuation limits are considered and the CBF is trained to approximate the maximal control invariant set inside the safe region. The details of this learning method are reported in the following paper:

[Learning Performance-Oriented Control Barrier Functions Under Complex Safety Constraints and Limited Actuation](https://arxiv.org/abs/2401.05629) \
Shaoru Chen, Mahyar Fazlyab
(Submitted to L4DC 2024)



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
