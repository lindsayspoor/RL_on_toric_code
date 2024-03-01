from evaluation_functions_dynamic import evaluate_error_rates, evaluate_fixed_errors
from plot_functions import plot_single_box_dynamic
from PPO_dynamic_agent import PPO_agent


# SETTINGS FOR RUNNING THIS SCRIPT

train = False # if True the agent will be trained, if False the agent will be loaded given the specified settings below. Please specify the storing folder in 'PPO_dynamic_agent.py'
log = True # if set to True the learning curve during training is registered and saved. Please specify the storing folder in 'PPO_dynamic_agent.py'
fixed = True # if set to True the agent is trained on training examples with a fixed amount of N initial errors. If set to False the agent is trained on training examples given an error rate 'error_rate' for each qubit to have a chance to be flipped.
evaluate_fixed = True # if set to True the trained model is evaluated on examples with a fixed amount of N initial errors. If set to False the trained model is evaluated on examples in which each qubit is flipped with a chance of 'error_rate'.

board_size = 3 # board of size dxd plaquettes and stars
error_rate = 0.1 # bit-flip error rate each qubit on the board is subject to
ent_coef = 0.05 # entropy coefficient of the model
clip_range = 0.1 # clipping parameter of the model
N = 1 # the number of fixed initinal flips N the agent model is trained on or loaded when 'fixed' is set to True.
new_N = 1 # the number of fixed new flips N_new after every k'th timestep (iteration_step).
iteration_step = 2 # number of timesteps before new errors are introduced.
logical_error_reward = 1 # the reward the agent gets when there is a logical error and it is game over.
empty_reward = 10 # the reward the agent gets when the board has no syndromes.
continue_reward = 1 # the reward the agent gets for each action that does not result in the terminal or empty board state. 
total_timesteps = 3000000 # total amount of times the env.step() is called during training. Note this is not equal to number of training episodes!
learning_rate = "annealing" # learning rate during training. If set to "annealing", the learning rate will go from 0.001 to 0.0001. Specify the begin and end values in 'PPO_dynamic_agent.py'. If set to a float, the learning rate is fixed to this value.

evaluate = True # if False, the agent won't be evaluated. If True, the agent will be evaluated.
render = False # if True, the environment with the agent's actions will be rendered per timestep.
save_files = True # if True results will be saved. Please specify the storing folder in the file 'evaluation_functions_static.py'
number_evaluations = 10000 # the number of evaluations the agent will be evaluated on
max_moves = 300 # the maximum amount of moves the agent is allowed to make per evaluation episode



# SET SETTINGS TO INITIALISE AGENT ON
initialisation_settings = {'board_size': board_size,
            'error_rate': error_rate,
            'lr':learning_rate,
            'total_timesteps': total_timesteps,
            'n_steps':2048,
            'mask_actions': True,
            'fixed':fixed,
            'c_reward':continue_reward,
            'e_reward':empty_reward,
            'l_reward':logical_error_reward,
            'N':N,
            'iteration_step': iteration_step,
            'ent_coef':ent_coef,
            'clip_range':clip_range,
            'new_N':new_N
            }

# SET SETTINGS TO LOAD TRAINED AGENT ON
loaded_model_settings = {'board_size': board_size,
            'error_rate': error_rate,
            'lr':learning_rate,
            'total_timesteps': total_timesteps,
            'n_steps':2048,
            'mask_actions': True,
            'fixed':fixed,
            'c_reward':continue_reward,
            'e_reward':empty_reward,
            'l_reward':logical_error_reward,
            'N':N,
            'iteration_step': iteration_step,
            'ent_coef':ent_coef,
            'clip_range':clip_range,
            'new_N':new_N
            }

# SET SETTINGS TO EVALUATE AGENT ON
evaluation_settings = {'board_size': board_size,
            'error_rate': error_rate,
            'lr':learning_rate,
            'total_timesteps': total_timesteps,
            'n_steps':2048,
            'mask_actions': True,
            'fixed':fixed,
            'c_reward':continue_reward,
            'e_reward':empty_reward,
            'l_reward':logical_error_reward,
            'N':N,
            'iteration_step': iteration_step,
            'ent_coef':ent_coef,
            'clip_range':clip_range,
            'new_N':new_N
            }




save_model_path = ''
for key, value in initialisation_settings.items():
    save_model_path += f"{key}={value}"


load_model_path = ''
for key, value in loaded_model_settings.items():
    load_model_path += f"{key}={value}"




#initialise PPO Agent
AgentPPO = PPO_agent(initialisation_settings, log)


if train:
    AgentPPO.train_model(save_model_path = save_model_path)
else:
    print(f"{loaded_model_settings['N']=}")
    print(f"{loaded_model_settings['error_rate']=}")
    AgentPPO.load_model(load_model_path = load_model_path)



        

if evaluate:

    if evaluate_fixed:
        mean_reward, rewards_agent, moves_agent,actions = evaluate_fixed_errors(AgentPPO, fixed, evaluation_settings, loaded_model_settings, render, number_evaluations, max_moves,  save_files)
    else:
        mean_reward, rewards_agent, moves_agent,actions = evaluate_error_rates(AgentPPO, fixed, evaluation_settings,  loaded_model_settings, render, number_evaluations, max_moves,  save_files)





evaluation_path = ''
for key, value in evaluation_settings.items():
    evaluation_path += f"{key}={value}"


if fixed:
    path = f"/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Figure_results/Results_benchmarks/boxplot_dynamic_ppo_{evaluation_path}_{loaded_model_settings['N']}.pdf"
else:
    path = f"/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Figure_results/Results_benchmarks/boxplot_dynamic_ppo_{evaluation_path}_{loaded_model_settings['error_rate']}.pdf"


plot_single_box_dynamic(path, rewards_agent)


