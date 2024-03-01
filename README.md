# RL on the toric code

This repository is constructed to run experiments for the thesis "Quantum error correction on the toric code using two distinct reinforcement learning game frameworks", and was originally made as a deliverable for the MSc Thesis of Lindsay Spoor. The research contains the training and evaluation of RL agents on the toric code decoding problem. The abstract can be found below.


### Abstract

This project employs reinforcement learning techniques to explore novel  decoding strategies for quantum error correction, particularly focusing on the toric code, to address the challenge of maintaining stable quantum states for fault-tolerant quantum computing. Two game frameworks are established, including a novel dynamic game framework applicable to the training and measuring of RL agents and potential application in multi-agent scenarios. The RL agents use Stable Baselines 3’s Proximal Policy Optimization and show to achieve Minimum Weight Perfect Matching performance on 3 × 3 toric code lattices in both the static and dynamic game frameworks.


## Requirements

The code for this project requires Python 3 and the following packages:
- Stable baselines 3
- OpenAI Gymnasium
- Numpy
- Matplotlib
- Pytorch


## Environments

This repository contains different environment files, all wrapped in OpenAI Gym, and were originally created for "A NEAT Quantum Error Decoder" (https://github.com/condensedAI/neat-qec), and are tailored to the purpose of this research. The following files contain environments:

- toric_game_static_env.py
	- Contains the environment used for the static game framework using a PPO or DQN agent with an MLP structure. The file contains the following environments:
		- ToricGameEnv: introduces uncorrelated bit-flip errors with an error rate
		- ToricGameFixedErrs: introduces uncorrelated bit-flip errors with a fixed amount of errors
		- ToricGameLocalErrs: introduces correlated bit-flip errors with an error rate
		- Board: Implementation of toric code on a board
- toric_game_static_env_cnn.py
	- Contains the environment used for the static game framework using a PPO or DQN agent with a CNN structure. The file contains the following environments:
		- ToricGameEnvCNN: introduces uncorrelated bit-flip errors with an error rate
		- ToricGameFixedErrsCNN: introduces uncorrelated bit-flip errors with a fixed amount of errors
		- Board: Implementation of toric code on a board
- toric_game_static_env_extra_action.py
	- Contains the translation of action space using an agent trained on a dynamic environment to be evaluated on a static environment. 
- toric_game_dynamic_env.py
	- Contains the environment used for the dynamic game framework using a PPO agent with an MLP structure. The file contains the following environments:
		- ToricGameDynamicEnv: introduces uncorrelated bit-flip errors with an error rate
		- ToricGameDynamicEnvFixedErrs: introduces uncorrelated bit-flip errors with a fixed amount of errors
		- Board: Implementation of toric code on a board


## Agents

The repository contains different files to initialise RL agents with, using Stable Baselines 3 as a library to train and evaluate the agents with. The following files are called to initialise agents with:

- PPO_static_agent.py
	- Contains a class that constructs, trains, loads a PPO agent with an MLP network architecture and is dependent on the environments stored in toric_game_static_env.py
- PPO_CNN_static_agent.py
	- Contains a class that constructs, trains, loads a PPO agent with an CNN network architecture and is dependent on the environments stored in toric_game_static_env_cnn.py
- DQN_static_agent.py
	- Contains a class that constructs, trains, loads a DQN agent with an MLP network architecture and is dependent on the environments stored in toric_game_static_env.py
- PPO_dynamic_agent.py
	- Contains a class that constructs, trains, loads a PPO agent with an MLP network architecture and is dependent on the environments stored in toric_game_dynamic_env.py


## Training and evaluating the agents

The repository contains different files to train and evaluate RL agents on the toric code. The following files can be executed from the command prompt:

- run_PPO_static_agent.py
	- Specify all settings in this file in order to either train or load and evaluate a PPO model on the static environment, and make sure the correct storing location is specified in PPO_static_agent.py.
- run_PPO_CNN_static_agent.py
	- Specify all settings in this file in order to either train or load and evaluate a PPO model with a CNN network architecture on the static environment, and make sure the correct storing location is specified in PPO_CNN_static_agent.py.
- run_DQN_static_agent.py
	- Specify all settings in this file in order to either train or load and evaluate a DQN model on the static environment, and make sure the correct storing location is specified in DQN_static_agent.py.
- run_PPO_dynamic_agent.py
	- Specify all settings in this file in order to either train or load and evaluate a PPO model on the dynamic environment, and make sure the correct storing location is specified in PPO_dynamic_agent.py.
