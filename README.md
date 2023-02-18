# Sim2Real2
Code for the ICRA 2023 paper "Sim2Real$^2$"

[Website]()|[arXiv]()|[Video]()

## About This Repo
We reimplemented the data generation code of Where2act with Sapien2.

We reimplemented the data generation code of Ditto with Sapien2 and Open3d. We collect data in a multi-processes manner like where2act.

For the Sampling-based Model Predictive Control module, we choose [ManiSkill2](https://github.com/haosulab/ManiSkill2)

### Installation

You should install Pointnet2_Pytorch as described in Where2act.

#### Ditto
2. Build ConvONets dependents by running `python scripts/convonet_setup.py build_ext --inplace`.

#### CEM
Install ManiSkill2 and ManiSkill2-Learn
```
cd {parent_directory_of_Sim2Real2}
cd Sim2Real2/CEM
pip install -e .
cd ManiSkill2-Learn/
pip install -e .
```

We integrate all requirements in `Sim2Real2/requirements.txt`

It should be noted that there are version conflicts between where2act and Ditto.
When training the where2act network, you should use (demanded by Pointnet2_Pytorch):
```
pytorch-lightning==0.7.1
hydra-core==0.11.3
```
When training the Ditto network, you should use:
```
pytorch-lightning==1.5.4
hydra-core==1.1.0.rc1
```

VHACD

### Simulation

#### CEM
```
cd {parent_directory_of_Sim2Real2}
cd Sim2Real2/CEM/ManiSkill2-Learn
python maniskill2_learn/apis/run_rl.py configs/mpc/cem/maniskill2_DigitalTwin.py --gpu-ids 0
```


### Real Experiments
In order to do the real experiment, a module that can acquire measurements from the depth camera and control the robot to replay the trajectory from where2act or CEM should be created additionally.

## Citations