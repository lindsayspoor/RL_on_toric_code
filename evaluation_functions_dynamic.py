import numpy as np
from tqdm import tqdm
from sb3_contrib.common.maskable.utils import get_action_masks


storing_folder = "/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/dynamic_ppo"


def evaluate_model(agent, evaluation_settings, render, number_evaluations, max_moves):
    
    print("evaluating the model...")
    moves = 0

    actions = np.zeros((number_evaluations,max_moves,1)) #1st column for agent (3rd dimension)
    actions[:,:,:] = np.nan
    reward_agent = []
    moves_agent = []


    for k in tqdm(range(number_evaluations)):
        rewards = 0
        moves = 0
        obs, info = agent.env.reset()

        if render:
            agent.env.render()
        for i in range(max_moves):
            if evaluation_settings['mask_actions']:
                action_masks = get_action_masks(agent.env)

                action, _state = agent.model.predict(obs, action_masks = action_masks)

            else:
                action, _state = agent.model.predict(obs)
            obs, reward, done, truncated, info = agent.env.step(action)

            actions[k,i,0] = action
            moves += 1
            rewards += reward
            if render:
                print(info['message'])
                agent.env.render()
            if done or truncated:
                break

        reward_agent.append(rewards)
        moves_agent.append(moves)


    mean_reward = np.mean(reward_agent)

    print("evaluation done")

    return mean_reward, reward_agent, moves_agent,actions



def evaluate_fixed_errors(agent, fixed, evaluation_settings, loaded_model_settings,render, number_evaluations, max_moves,save_files):
    

    mean_reward_agent, reward_agent, moves_agent,actions = evaluate_model(agent,evaluation_settings, render, number_evaluations, max_moves)

    evaluation_path = ''
    for key, value in evaluation_settings.items():
        evaluation_path += f"{key}={value}"

    if save_files:
        if fixed:
            np.savetxt(f"{storing_folder}/rewards_dynamic_agent/rewards_dynamic_ppo_{evaluation_path}_{loaded_model_settings['N']}.csv", reward_agent)
            np.savetxt(f"{storing_folder}/moves_dynamic_agent/moves_dynamic_ppo_{evaluation_path}_{loaded_model_settings['N']}.csv", moves_agent)

        else:
            np.savetxt(f"{storing_folder}/rewards_dynamic_agent/rewards_dynamic_ppo_{evaluation_path}_{loaded_model_settings['error_rate']}.csv", reward_agent)
            np.savetxt(f"{storing_folder}/moves_dynamic_agent/moves_dynamic_ppo_{evaluation_path}_{loaded_model_settings['error_rate']}.csv", moves_agent)


    return mean_reward_agent, reward_agent, moves_agent,actions


def evaluate_error_rates(agent, fixed, evaluation_settings, loaded_model_settings,render, number_evaluations, max_moves, save_files):
    

    mean_reward_agent, reward_agent, moves_agent,actions = evaluate_model(agent,evaluation_settings, render, number_evaluations, max_moves)

    evaluation_path = ''
    for key, value in evaluation_settings.items():
        evaluation_path += f"{key}={value}"


    if save_files:
        if fixed:
            np.savetxt(f"{storing_folder}/rewards_dynamic_agent/rewards_dynamic_ppo_{evaluation_path}_{loaded_model_settings['N']}.csv", reward_agent)
            np.savetxt(f"{storing_folder}/moves_dynamic_agent/moves_dynamic_ppo_{evaluation_path}_{loaded_model_settings['N']}.csv", moves_agent)

        else:
            np.savetxt(f"{storing_folder}/rewards_dynamic_agent/rewards_dynamic_ppo_{evaluation_path}_{loaded_model_settings['error_rate']}.csv", reward_agent)
            np.savetxt(f"{storing_folder}/moves_dynamic_agent/moves_dynamic_ppo_{evaluation_path}_{loaded_model_settings['error_rate']}.csv", moves_agent)



    return mean_reward_agent, reward_agent, moves_agent, actions

