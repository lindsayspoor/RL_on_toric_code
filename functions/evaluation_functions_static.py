import numpy as np
from functions.MWPM_decoder import decode_MWPM_pymatching
from tqdm import tqdm
from sb3_contrib.common.maskable.utils import get_action_masks
from functions.plot_functions import render_evaluation
from environments.toric_game_static_env_extra_action import ToricGameEnvExtraAction, ToricGameEnvExtraActionFixed



storing_folder = "/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/static_ppo"
storing_folder_dynamic_on_static = "/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/static_vs_dynamic_ppo"


def evaluate_model(agent, evaluation_settings, render, number_evaluations, max_moves, check_fails):

    print("evaluating the model...")

    moves = 0
    logical_errors = 0
    max_reached = 0
    success = 0
    success_MWPM = 0
    logical_errors_MWPM = 0

    observations = np.zeros((number_evaluations, evaluation_settings['board_size']*evaluation_settings['board_size']))
    results = np.zeros((number_evaluations,2)) # 1st column for agent, 2nd column for MWPM decoder
    actions = np.zeros((number_evaluations,max_moves,2)) # 1st column for agent, 2nd column for MWPM decoder (3rd dimension)
    actions[:,:,:] = np.nan
    


    for k in tqdm(range(number_evaluations)):

        obs, info = agent.env.reset(allow_empty=True)
        initial_flips = agent.env.initial_qubits_flips
        if render:
            agent.env.render()
        obs0 = obs.copy()
        observations[k,:] = obs
        obs0_k = obs0.reshape((evaluation_settings['board_size'],evaluation_settings['board_size']))

        MWPM_check, MWPM_actions = decode_MWPM_pymatching(agent.env.parity_check_matrix_plaqs,agent.env.state.qubit_pos,obs0, initial_flips, evaluation_settings)

        actions[k,:MWPM_actions.shape[0],1] = MWPM_actions[:,0]

        if MWPM_check:
            success_MWPM += 1
            results[k,1] = 1 # 1 for success
        if not MWPM_check:
            logical_errors_MWPM += 1
            results[k,1] = 0 # 0 for fail


        for i in range(max_moves):
            if not agent.env.done:
                if evaluation_settings['mask_actions']:
                    action_masks = get_action_masks(agent.env)
                    action, _state = agent.model.predict(obs, action_masks=action_masks, deterministic=True)
                else:
                    action, _state = agent.model.predict(obs)
            else:
                action = None
            obs, reward, done, truncated, info = agent.env.step(action)
            actions[k,i,0] = action



            moves += 1
            if render:
                agent.env.render()
            if done:
                if info['message'] == 'logical_error':
                    if check_fails:
                        if results[k,0] == 1 and results[k,1] == 0:
                            
                            print(info['message'])
                            render_evaluation(obs0_k,evaluation_settings, actions[k,:,:], initial_flips)

                    logical_errors += 1
                    results[k,0] = 0 # 0 for fail
                if info['message'] == 'success':
                    success += 1
                    results[k,0] = 1 # 1 for success

                break

        if not done:
            max_reached += 1




        

    print(f"mean number of moves per evaluation is {moves/number_evaluations}")
    
    if (success+logical_errors) == 0:
        success_rate = 0
    else:
        success_rate = success / (success+logical_errors+max_reached)


    if (success_MWPM+logical_errors_MWPM) == 0:
        success_rate_MWPM = 0
    else:
        success_rate_MWPM = success_MWPM / (success_MWPM+logical_errors_MWPM)
    

    print("evaluation done")


    return success_rate, success_rate_MWPM, observations, results, actions



def evaluate_fixed_errors(agent, evaluate_fixed, fixed, evaluation_settings, loaded_model_settings,N_evaluates, render, number_evaluations, max_moves, check_fails, save_files):
    
    success_rates = []
    success_rates_MWPM = []
    observations_all = []

    for N_evaluate in N_evaluates:
        print(f"{N_evaluate=}")
        evaluation_settings['fixed'] = evaluate_fixed
        evaluation_settings['N'] = N_evaluate
        evaluation_settings['success_reward'] = evaluation_settings['N']
        agent.change_environment_settings(evaluation_settings)
        success_rate, success_rate_MWPM, observations, results, actions = evaluate_model(agent,evaluation_settings, render, number_evaluations, max_moves, check_fails)
        success_rates.append(success_rate)
        success_rates_MWPM.append(success_rate_MWPM)
        observations_all.append(observations)
        print(f"{success_rate=}")
        print(f"{success_rate_MWPM=}")



    success_rates = np.array(success_rates)
    success_rates_MWPM = np.array(success_rates_MWPM)
    observations_all = np.array(observations_all)
    print(f"{observations_all.shape=}")


    evaluation_path = ''
    for key, value in evaluation_settings.items():
        evaluation_path += f"{key}={value}"

    if save_files:
        if fixed:
            np.savetxt(f"{storing_folder}/success_rates_agent/success_rates_{evaluation_path}_{loaded_model_settings['N']}.csv", success_rates)
            np.savetxt(f"{storing_folder}/success_rates_MWPM/success_rates_{evaluation_path}_{loaded_model_settings['N']}.csv", success_rates_MWPM)
            np.savetxt(f"{storing_folder}/observations/observations_{evaluation_path}_{loaded_model_settings['N']}.csv", observations)
            np.savetxt(f"{storing_folder}/results_agent_MWPM/results_{evaluation_path}_{loaded_model_settings['N']}.csv", results)
            np.savetxt(f"{storing_folder}/actions_agent/actions_agent_{evaluation_path}_{loaded_model_settings['N']}.csv", actions[:,:,0])
            np.savetxt(f"{storing_folder}/actions_MWPM/actions_MWPM_{evaluation_path}_{loaded_model_settings['N']}.csv", actions[:,:,1])
        else:
            np.savetxt(f"{storing_folder}/success_rates_agent/success_rates_{evaluation_path}_{loaded_model_settings['error_rate']}.csv", success_rates)
            np.savetxt(f"{storing_folder}/success_rates_MWPM/success_rates_{evaluation_path}_{loaded_model_settings['error_rate']}.csv", success_rates_MWPM)
            np.savetxt(f"{storing_folder}/observations/observations_{evaluation_path}_{loaded_model_settings['error_rate']}.csv", observations)
            np.savetxt(f"{storing_folder}/results_agent_MWPM/results_{evaluation_path}_{loaded_model_settings['error_rate']}.csv", results)
            np.savetxt(f"{storing_folder}/actions_agent/actions_agent_{evaluation_path}_{loaded_model_settings['error_rate']}.csv", actions[:,:,0])
            np.savetxt(f"{storing_folder}/actions_MWPM/actions_MWPM_{evaluation_path}_{loaded_model_settings['error_rate']}.csv", actions[:,:,1])

    return success_rates, success_rates_MWPM


def evaluate_error_rates(agent, evaluate_fixed, fixed, evaluation_settings, loaded_model_settings, error_rates, render, number_evaluations, max_moves, check_fails, save_files):
    success_rates = []
    success_rates_MWPM = []
    observations_all = []

    for error_rate in error_rates:
        # SET SETTINGS TO EVALUATE LOADED AGENT ON
        print(f"{error_rate=}")
        evaluation_settings['error_rate'] = error_rate
        evaluation_settings['fixed'] = evaluate_fixed

        agent.change_environment_settings(evaluation_settings)
        success_rate, success_rate_MWPM, observations, results, actions = evaluate_model(agent,evaluation_settings, render, number_evaluations, max_moves, check_fails)
        success_rates.append(success_rate)
        success_rates_MWPM.append(success_rate_MWPM)
        observations_all.append(observations)
        print(f"{success_rate=}")
        print(f"{success_rate_MWPM=}")



    success_rates = np.array(success_rates)
    success_rates_MWPM = np.array(success_rates_MWPM)
    observations_all = np.array(observations_all)



    evaluation_path = ''
    for key, value in evaluation_settings.items():
        evaluation_path += f"{key}={value}"

    if save_files:
        if fixed:
            np.savetxt(f"{storing_folder}/success_rates_agent/success_rates_{evaluation_path}_{loaded_model_settings['N']}.csv", success_rates)
            np.savetxt(f"{storing_folder}/success_rates_MWPM/success_rates_{evaluation_path}_{loaded_model_settings['N']}.csv", success_rates_MWPM)
            np.savetxt(f"{storing_folder}/observations/observations_{evaluation_path}_{loaded_model_settings['N']}.csv", observations)
            np.savetxt(f"{storing_folder}/results_agent_MWPM/results_{evaluation_path}_{loaded_model_settings['N']}.csv", results)
            np.savetxt(f"{storing_folder}/actions_agent/actions_agent_{evaluation_path}_{loaded_model_settings['N']}.csv", actions[:,:,0])
            np.savetxt(f"{storing_folder}/actions_MWPM/actions_MWPM_{evaluation_path}_{loaded_model_settings['N']}.csv", actions[:,:,1])
        else:
            np.savetxt(f"{storing_folder}/success_rates_agent/success_rates_{evaluation_path}_{loaded_model_settings['error_rate']}.csv", success_rates)
            np.savetxt(f"{storing_folder}/success_rates_MWPM/success_rates_{evaluation_path}_{loaded_model_settings['error_rate']}.csv", success_rates_MWPM)
            np.savetxt(f"{storing_folder}/observations/observations_{evaluation_path}_{loaded_model_settings['error_rate']}.csv", observations)
            np.savetxt(f"{storing_folder}/results_agent_MWPM/results_{evaluation_path}_{loaded_model_settings['error_rate']}.csv", results)
            np.savetxt(f"{storing_folder}/actions_agent/actions_agent_{evaluation_path}_{loaded_model_settings['error_rate']}.csv", actions[:,:,0])
            np.savetxt(f"{storing_folder}/actions_MWPM/actions_MWPM_{evaluation_path}_{loaded_model_settings['error_rate']}.csv", actions[:,:,1])

    return success_rates, success_rates_MWPM



def evaluate_error_rates_dynamic_on_static(agent, fixed, evaluation_settings, loaded_model_settings, error_rates, render, number_evaluations, max_moves, check_fails, save_files):
    success_rates = []
    success_rates_MWPM = []
    observations_all = []

    for error_rate in error_rates:
        # SET SETTINGS TO EVALUATE LOADED AGENT ON
        print(f"{error_rate=}")
        evaluation_settings['error_rate'] = error_rate

        change_environment_settings_dynamic_vs_static(agent, evaluation_settings)
        success_rate, success_rate_MWPM, observations, results, actions = evaluate_model(agent,evaluation_settings, render, number_evaluations, max_moves, check_fails)
        success_rates.append(success_rate)
        success_rates_MWPM.append(success_rate_MWPM)
        observations_all.append(observations)
        print(f"{success_rate=}")
        print(f"{success_rate_MWPM=}")



    success_rates = np.array(success_rates)
    success_rates_MWPM = np.array(success_rates_MWPM)
    observations_all = np.array(observations_all)



    evaluation_path = ''
    for key, value in evaluation_settings.items():
        evaluation_path += f"{key}={value}"

    load_model_path = ''
    for key, value in loaded_model_settings.items():
        load_model_path += f"{key}={value}"

    if save_files:
        if fixed:
            np.savetxt(f"{storing_folder_dynamic_on_static}/success_rates_agent/success_rates_{load_model_path}_{loaded_model_settings['N']}.csv", success_rates)
            np.savetxt(f"{storing_folder_dynamic_on_static}/success_rates_MWPM/success_rates_{load_model_path}_{loaded_model_settings['N']}.csv", success_rates_MWPM)
            np.savetxt(f"{storing_folder_dynamic_on_static}/observations/observations_{load_model_path}_{loaded_model_settings['N']}.csv", observations)
            np.savetxt(f"{storing_folder_dynamic_on_static}/results_agent_MWPM/results_{load_model_path}_{loaded_model_settings['N']}.csv", results)
            np.savetxt(f"{storing_folder_dynamic_on_static}/actions_agent/actions_agent_{load_model_path}_{loaded_model_settings['N']}.csv", actions[:,:,0])
            np.savetxt(f"{storing_folder_dynamic_on_static}/actions_MWPM/actions_MWPM_{load_model_path}_{loaded_model_settings['N']}.csv", actions[:,:,1])
        else:
            np.savetxt(f"{storing_folder_dynamic_on_static}/success_rates_agent/success_rates_{load_model_path}_{loaded_model_settings['error_rate']}.csv", success_rates)
            np.savetxt(f"{storing_folder_dynamic_on_static}/success_rates_MWPM/success_rates_{load_model_path}_{loaded_model_settings['error_rate']}.csv", success_rates_MWPM)
            np.savetxt(f"{storing_folder_dynamic_on_static}/observations/observations_{load_model_path}_{loaded_model_settings['error_rate']}.csv", observations)
            np.savetxt(f"{storing_folder_dynamic_on_static}/results_agent_MWPM/results_{load_model_path}_{loaded_model_settings['error_rate']}.csv", results)
            np.savetxt(f"{storing_folder_dynamic_on_static}/actions_agent/actions_agent_{load_model_path}_{loaded_model_settings['error_rate']}.csv", actions[:,:,0])
            np.savetxt(f"{storing_folder_dynamic_on_static}/actions_MWPM/actions_MWPM_{load_model_path}_{loaded_model_settings['error_rate']}.csv", actions[:,:,1])

    return success_rates, success_rates_MWPM


def change_environment_settings_dynamic_vs_static(agent, settings):
    print("changing environment settings...")
    if settings['fixed']:
        agent.env = ToricGameEnvExtraActionFixed(settings)
    else:
        agent.env = ToricGameEnvExtraAction(settings)

