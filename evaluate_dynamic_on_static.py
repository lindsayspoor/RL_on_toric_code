import numpy as np
from plot_functions import plot_benchmark_MWPM
from PPO_dynamic_agent import PPO_agent
from evaluation_functions_static import evaluate_error_rates_dynamic_on_static




# SETTINGS FOR DYNAMIC ENV

train = False # if True the agent will be trained, if False the agent will be loaded given the specified settings below. Please specify the storing folder in 'PPO_dynamic_agent.py'
log = True # if set to True the learning curve during training is registered and saved. Please specify the storing folder in 'PPO_dynamic_agent.py'
fixed = True # if set to True the agent is trained on training examples with a fixed amount of N initial errors. If set to False the agent is trained on training examples given an error rate 'error_rate' for each qubit to have a chance to be flipped.

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




#SET SETTINGS TO INITIALISE AGENT ON
dynamic_initialisation_settings = {'board_size': board_size,
            'error_rate': error_rate,
            'lr':learning_rate,
            'total_timesteps': total_timesteps,
            'n_steps':2048,
            'mask_actions': True,
            'fixed':fixed,
            'c_reward':continue_reward,
            'e_reward':empty_reward,
            'l_reward':logical_error_reward,
            'N':N,#,
            'iteration_step': iteration_step,
            'ent_coef':ent_coef,
            'clip_range':clip_range,
            'new_N':new_N
            }

#SET SETTINGS TO LOAD TRAINED AGENT ON
dynamic_loaded_model_settings = {'board_size': board_size,
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



dynamic_save_model_path = ''
for key, value in dynamic_initialisation_settings.items():
    dynamic_save_model_path += f"{key}={value}"


dynamic_load_model_path = ''
for key, value in dynamic_loaded_model_settings.items():
    dynamic_load_model_path += f"{key}={value}"




#initialise dynamic PPO Agent
AgentPPO = PPO_agent(dynamic_initialisation_settings, log)
AgentPPO.load_model(load_model_path = dynamic_load_model_path)


# SETTINGS FOR STATIC ENV

logical_error_reward = 5 # the reward the agent gets when it has removed all syndrome points, but the terminal board state claims that there is a logical error.
success_reward = 10 # the reward the agent gets when it has removed all syndrome points, and the terminal board state claims that there is no logical error, ans therefore the agent has successfully done its job.
continue_reward = -1 # the reward the agent gets for each action that does not result in the terminal board state. If negative it gets penalized for each move it does, therefore giving the agent an incentive to remove syndromes in as less moves as possible.
illegal_action_reward = -2 # the reward the agent gets when 'mask_actions' is set to False and therefore the agent gets penalized by choosing an illegal action.

correlated = False # if True, the agent will be evaluated on an environment providing correlated bit-flip errors. If False, the environment will introduce uncorrelated bit-flip errors.
N_evaluates = [1,2,3,4,5] # the number of fixed initial flips N the agent is evaluated on if 'evaluate_fixed' is set to True.
error_rates_eval = list(np.linspace(0.01,0.15,10)) # the error rates the agent is evaluated on if 'evaluate_fixed' is set to False.


evaluate = True # if False, the agent won't be evaluated. If True, the agent will be evaluated.
render = False # if True, the environment with the agent's actions will be rendered per timestep.
save_files = True # if True results will be saved. Please specify the storing folder in the file 'evaluation_functions_static.py'
number_evaluations = 10000 # the number of evaluations the agent will be evaluated on
max_moves = 300 # the maximum amount of moves the agent is allowed to make per evaluation episode
check_fails = False # if True, during evaluation all cases in which the agent fails, but MWPM succeeds, will be rendered.



#SET SETTINGS TO EVALUATE STATIC ENVIROMNENT ON

static_evaluation_settings = {'board_size': board_size,
            'error_rate': error_rate,
            'l_reward': logical_error_reward,
            's_reward': success_reward,
            'c_reward':continue_reward,
            'i_reward':illegal_action_reward,
            'lr':learning_rate,
            'total_timesteps': total_timesteps,
            'mask_actions':  True,
            'correlated':correlated,
            'fixed':fixed,
            'N':N,
            'ent_coef':ent_coef,
            'clip_range':clip_range,
            'max_moves':max_moves
            }




if evaluate:
    success_rates_all, success_rates_all_MWPM,observations, results, actions = evaluate_error_rates_dynamic_on_static(AgentPPO, fixed, static_evaluation_settings, dynamic_loaded_model_settings, error_rates_eval, render, number_evaluations, max_moves, check_fails, save_files)


success_rates_all = np.array(success_rates_all)
success_rates_all_MWPM = np.array(success_rates_all_MWPM)



if fixed:
    path_plot = f"/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Figure_results/Results_benchmarks/dynamic_vs_MWPM_{dynamic_load_model_path}_{dynamic_loaded_model_settings['N']}.pdf"
else:
    path_plot = f"/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Figure_results/Results_benchmarks/dynamic_vs_MWPM_{dynamic_load_model_path}_{dynamic_loaded_model_settings['error_rate']}.pdf"


plot_benchmark_MWPM(success_rates_all, success_rates_all_MWPM, N_evaluates, error_rates_eval, board_size,path_plot,False)



