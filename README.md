# Sample_RL

This repo will contain sample code for making use of GPU to run machine learning algorithms and sample RL codes using with and without DQN

# Setting up the environment 

Install anaconda using the instructions in this [link](https://docs.anaconda.com/anaconda/install/linux/)

Create an environment and install python 3.7 if not already installed

```
conda create -n rl_intro
conda activate rl_intro
conda install python=3.7
```

Then install openAI gym using the following:

```
git clone https://github.com/openai/gym
cd gym
pip install -e .
```

Using `pip install -e .[all]` requires MuJoCo to be installed which I did not install so I ignored it.


