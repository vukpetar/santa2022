# Santa 2022
The goal of this project is to create a Reinforcement Learning Agent that will solve the optimization task of the [Santa 2022 - The Christmas Card Conundrum](https://www.kaggle.com/competitions/santa-2022) kaggle competition.

# Requirements
In order to be able to run the code found in this repository, it is necessary to:
* Python interpreter (>=3.8.0) (Recommendation: [Anaconda](https://www.anaconda.com/products/distribution))
* Python packages:
    * NumPy
    * SciPy
    * Matplotlib
    * [Pytorch](https://pytorch.org/)
    * [stable-baselines3](https://pypi.org/project/stable-baselines3/)

# How to start
1. ### Clone the repository
```
git clone https://github.com/vukpetar/santa2022.git
```
2. ### Change the repo
```
cd santa2022
```
3. ### Download Anaconda from [this link](https://www.anaconda.com/products/distribution) and install
4. ### Install conda environment
On Windows:
```
conda env create -f environments/santa2022_env_windows.yml
```
On Linux:
```
conda env create -f environments/santa2022_env_windows.yml
```
5. ### Activate conda environment
On Windows:
```
conda activate santa2022
```
On Linux and macOSndows:
```
source activate santa2022

# Description

The code in this repository consists of two python files and one Jupyter Notebook:

* **utils.py** - Helper fuctions for Santa2022 task.
* **environment.py** - Gym environment class.
* **main.ipynb** - Notebook for training the Proximal Policy Optimization (PPO) algorithm in the stable-baselines3 python package.

# TODO
| **Features**                             | **Status**         |
| ---------------------------------------- | -------------------|
| Create custom gym environment            | :heavy_check_mark: |
| Train PPO                                | :heavy_check_mark: |
| Create custom PPO algorithm              | :x:                |
| Create RL Agent as Graph Neural Network  | :x:                |