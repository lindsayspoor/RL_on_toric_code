from evaluation_functions_PPO_static import evaluate_error_rates, evaluate_fixed_errors
from plot_functions import plot_benchmark_MWPM
import numpy as np
from PPO_static_agent import PPO_agent



# SETTINGS FOR RUNNING THIS SCRIPT

train=True # if True the agent will be trained, if False the agent will be loaded given the specified settings below. Please specify the storing folder in 'PPO_static_agent.py'
curriculum=False #if False, curriculum learning on multiple sequential error values is disabled. Please specify either 'training_error_rates' or 'training_N' as a list containing a single value.
# If True, curriculum is enabled. Please specify 'training_error_rates' of 'training_N' as a list containing multiple values to train on sequentially.
log = True # if set to True the learning curve during training is registered and saved. Please specify the storing folder in 'PPO_static_agent.py'
correlated=False # if True, the agent will be initialised on an environment providing correlated bit-flip errors. If False, the environment will introduce uncorrelated bit-flip errors.
fixed=False # if set to True the agent is trained on training examples with a fixed amount of N initial errors. If set to False the agent is trained on training examples given an error rate 'error_rate' for each qubit to have a chance to be flipped.
evaluate_fixed=False # if set to True the trained model is evaluated on examples with a fixed amount of N initial errors. If set to False the trained model is evaluated on examples in which each qubit is flipped with a chance of 'error_rate'.
mask_actions=True # if set to True action masking is enabled, the illegal actions are masked out by the model. If set to False the agent gets a reward 'illegal_action_reward' when choosing an illegal action.

board_size=5 # board of size dxd plaquettes and stars
error_rate=0.1 # bit-flip error rate each qubit on the board is subject to
ent_coef=0.01 # entropy coefficient of the model
clip_range=0.1 # clipping parameter of the model
N=1 # the number of fixed initinal flips N the agent model is trained on or loaded when 'fixed' is set to True
logical_error_reward=5 # the reward the agent gets when it has removed all syndrome points, but the terminal board state claims that there is a logical error.
success_reward=10 # the reward the agent gets when it has removed all syndrome points, and the terminal board state claims that there is no logical error, ans therefore the agent has successfully done its job.
continue_reward=-1 # the reward the agent gets for each action that does not result in the terminal board state. If negative it gets penalized for each move it does, therefore giving the agent an incentive to remove syndromes in as less moves as possible.
illegal_action_reward=-2 # the reward the agent gets when 'mask_actions' is set to False and therefore the agent gets penalized by choosing an illegal action.
total_timesteps=1000 # total amount of times the env.step() is called during training. Note this is not equal to number of training episodes!
learning_rate= 0.001 # learning rate during training

training_N=[N] # values of N initial flips the agent model is trained on
# training_error_rates=list(np.linspace(0.01,0.15,6))
training_error_rates=[error_rate] # values of error rates the agent model is trained on

evaluate=True # if False, the agent won't be evaluated. If True, the agent will be evaluated.
check_fails=False # if True, during evaluation all cases in which the agent fails, but MWPM succeeds, will be rendered.
render=False # if True, the environment with the agent's actions will be rendered per timestep.
save_files=True # if True results will be saved. Please specify the storing folder in the file 'evaluation_functions_PPO_static.py'
number_evaluations=100 # the number of evaluations the agent will be evaluated on
max_moves=200 # the maximum amount of moves the agent is allowed to make per evaluation episode
N_evaluates = [1,2,3,4,5] # the number of fixed initial flips N the agent is evaluated on if 'evaluate_fixed' is set to True.
error_rates_eval=list(np.linspace(0.01,0.15,10)) # the error rates the agent is evaluated on if 'evaluate_fixed' is set to False.



# SET SETTINGS TO INITIALISE AGENT ON
initialisation_settings = {'board_size': board_size,
            'error_rate': error_rate,
            'l_reward': logical_error_reward,
            's_reward': success_reward,
            'c_reward':continue_reward,
            'i_reward':illegal_action_reward,
            'lr':learning_rate,
            'total_timesteps': total_timesteps,
            'mask_actions': mask_actions,
            'correlated':correlated,
            'fixed':fixed,
            'N':N,
            'ent_coef':ent_coef,
            'clip_range':clip_range
            }

# SET SETTINGS TO LOAD TRAINED AGENT ON
loaded_model_settings = {'board_size': board_size,
            'error_rate': error_rate,
            'l_reward': logical_error_reward,
            's_reward': success_reward,
            'c_reward':continue_reward,
            'i_reward':illegal_action_reward,
            'lr':learning_rate,
            'total_timesteps': total_timesteps,
            'mask_actions': mask_actions,
            'correlated':correlated,
            'fixed':fixed,
            'N':N,
            'ent_coef':ent_coef,
            'clip_range':clip_range
            }

# SET SETTINGS TO EVALUATE AGENT ON
evaluation_settings = {'board_size': board_size,
            'error_rate': error_rate,
            'l_reward': logical_error_reward,
            's_reward': success_reward,
            'c_reward':continue_reward,
            'i_reward':illegal_action_reward,
            'lr':learning_rate,
            'total_timesteps': total_timesteps,
            'mask_actions': mask_actions,
            'correlated':correlated,
            'fixed':fixed,
            'N':N,
            'ent_coef':ent_coef,
            'clip_range':clip_range
            }




success_rates_all=[]
success_rates_all_MWPM=[]

if fixed:
    training_values=training_N
else:
    training_values=training_error_rates


for training_value in training_values:

    if (train==True) and (curriculum == False) and(training_values.index(training_value)>0):
        train=False
        curriculum=True


    save_model_path =''
    for key, value in initialisation_settings.items():
        save_model_path+=f"{key}={value}"


    load_model_path =''
    for key, value in loaded_model_settings.items():
        load_model_path+=f"{key}={value}"




    # initialise PPO Agent
    AgentPPO = PPO_agent(initialisation_settings, log)

    if train:
        AgentPPO.train_model(save_model_path=save_model_path)
    else:
        print(f"{loaded_model_settings['N']=}")
        print(f"{loaded_model_settings['error_rate']=}")
        AgentPPO.load_model(load_model_path=load_model_path)
        

    if curriculum:
        if fixed:
        
            print(f"{training_value=}")
            initialisation_settings['N']=training_value
        else:
            print(f"{training_value=}")
            initialisation_settings['error_rate']=training_value



        save_model_path =''
        for key, value in initialisation_settings.items():
            save_model_path+=f"{key}={value}"

        AgentPPO.change_environment_settings(initialisation_settings)

        AgentPPO.train_model(save_model_path=save_model_path)
        
        if fixed:
            loaded_model_settings['N']=training_value
        else:
            loaded_model_settings['error_rate']=training_value



    if evaluate:

        if evaluate_fixed:
            success_rates, success_rates_MWPM = evaluate_fixed_errors(AgentPPO,evaluate_fixed, fixed, evaluation_settings, loaded_model_settings,N_evaluates, render, number_evaluations, max_moves, check_fails, save_files)
        else:
            success_rates, success_rates_MWPM = evaluate_error_rates(AgentPPO, evaluate_fixed, fixed, evaluation_settings, loaded_model_settings,error_rates_eval, render, number_evaluations, max_moves, check_fails, save_files)


        success_rates_all.append(success_rates)
        success_rates_all_MWPM.append(success_rates_MWPM)



evaluation_path =''
for key, value in evaluation_settings.items():
    evaluation_path+=f"{key}={value}"



success_rates_all=np.array(success_rates_all)
success_rates_all_MWPM=np.array(success_rates_all_MWPM)


if fixed:
    path_plot = f"/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Figure_results/Results_benchmarks/PPO_vs_MWPM_{evaluation_path}_{loaded_model_settings['N']}.pdf"
else:
    path_plot = f"/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Figure_results/Results_benchmarks/PPO_vs_MWPM_{evaluation_path}_{loaded_model_settings['error_rate']}.pdf"


plot_benchmark_MWPM(success_rates_all, success_rates_all_MWPM, N_evaluates, error_rates_eval, board_size,path_plot,loaded_model_settings['N'], loaded_model_settings['error_rate'],evaluate_fixed)