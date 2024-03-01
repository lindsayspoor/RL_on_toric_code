# this file contains all the necessary plotting functions
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.results_plotter import load_results, ts2xy

def plot_benchmark_MWPM(success_rates_all, success_rates_all_MWPM,N_evaluates, error_rates_eval, board_size,path_plot,evaluate_fixed):

    plt.figure(figsize = (6,4))
    if evaluate_fixed:
        plt.plot(N_evaluates, success_rates_all_MWPM[-1,:], label=f'd={board_size} MWPM decoder', linestyle='-.', linewidth=0.5, color='black')
        plt.scatter(N_evaluates, success_rates_all[-1,:], label=f"d={board_size} PPO agent", marker="^", s=30)
        plt.plot(N_evaluates, success_rates_all[-1,:], linestyle='-.', linewidth=0.5)
        plt.xlabel(r'N')
    else:
        plt.plot(error_rates_eval, success_rates_all_MWPM[-1,:], label=f'd={board_size} MWPM', linestyle='-.',linewidth=0.9, color='black')
        plt.scatter(error_rates_eval, success_rates_all[-1,:], label=f"d={board_size} PPO agent", marker="^", s=40, color = 'blue')
        plt.plot(error_rates_eval, success_rates_all[-1,:], linestyle='-',linewidth=0.9, color='blue')
        plt.xlabel(r'$p$')
        plt.xlim((0,error_rates_eval[-1]+0.005))


    plt.ylabel(r'$p_s$')
    plt.legend()
    plt.grid()
    plt.savefig(path_plot)



def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window

    return np.convolve(values, weights, "valid")


def calculate_rolling(values, window):

    means = []
    errorbars = []
    for i in range(0, values.shape[0]):
        mean = np.mean(values[i:(i+window)])
        errors = np.std(values[i:(i+window)])
        means.append(mean)
        errorbars.append(errors)


    return np.array(means), np.array(errorbars)
    




def plot_log_results(log_folder,  save_model_path, title="Average training reward"):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), "timesteps")

    y = moving_average(y, window=50)

    # Truncate x
    x = x[len(x) - len(y) :]

    np.savetxt(f"/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/log_results/log_results_{save_model_path}.csv",(x,y) )

    fig = plt.figure(title)
    plt.plot(x, y, color = 'blue', linewidth=0.9)
    plt.yscale("linear")
    plt.xlabel("Number of training timesteps")
    plt.ylabel("Reward")
    plt.grid()
    plt.savefig(f'/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Figure_results/Results_reward_logs/learning_curve_{save_model_path}.pdf')



def render_evaluation(obs0_k,evaluation_settings, actions_k, initial_flips_k):
        size=evaluation_settings['board_size']
        qubit_pos   = [[x,y] for x in range(2*size) for y in range((x+1)%2, 2*size, 2)]
        plaquet_pos = [[x,y] for x in range(1,2*size,2) for y in range(1,2*size,2)]


        fig, (ax3,ax1,ax2) = plt.subplots(1,3, figsize=(15,5))
        a=1/(2*size)

        for i, p in enumerate(plaquet_pos):
            if obs0_k.flatten()[i]==1:

                fc='darkorange'
                plaq = plt.Polygon([[a*p[0], a*(p[1]-1)], [a*(p[0]+1), a*(p[1])], [a*p[0], a*(p[1]+1)], [a*(p[0]-1), a*p[1]] ], fc=fc)
                ax1.add_patch(plaq)

        for i, p in enumerate(plaquet_pos):
            if obs0_k.flatten()[i]==1:

                fc='darkorange'
                plaq = plt.Polygon([[a*p[0], a*(p[1]-1)], [a*(p[0]+1), a*(p[1])], [a*p[0], a*(p[1]+1)], [a*(p[0]-1), a*p[1]] ], fc=fc)
                ax2.add_patch(plaq)

        for i, p in enumerate(plaquet_pos):
            if obs0_k.flatten()[i]==1:

                fc='darkorange'
                plaq = plt.Polygon([[a*p[0], a*(p[1]-1)], [a*(p[0]+1), a*(p[1])], [a*p[0], a*(p[1]+1)], [a*(p[0]-1), a*p[1]] ], fc=fc)
                ax3.add_patch(plaq)

        # Draw lattice
        for x in range(size):
            for y in range(size):
                pos=(2*a*x, 2*a*y)
                width=a*2
                lattice = plt.Rectangle( pos, width, width, fc='none', ec='black' )
                ax1.add_patch(lattice)

        for x in range(size):
            for y in range(size):
                pos=(2*a*x, 2*a*y)
                width=a*2
                lattice = plt.Rectangle( pos, width, width, fc='none', ec='black' )
                ax2.add_patch(lattice)

        for x in range(size):
            for y in range(size):
                pos=(2*a*x, 2*a*y)
                width=a*2
                lattice = plt.Rectangle( pos, width, width, fc='none', ec='black' )
                ax3.add_patch(lattice)

        for i, p in enumerate(qubit_pos):
            pos=(a*p[0], a*p[1])
            fc1='darkgrey'
            if i in list(actions_k[:,0]):
                fc1 = 'darkblue'


            circle1 = plt.Circle( pos , radius=a*0.25, ec='k', fc=fc1)
            ax1.add_patch(circle1)
            ax1.annotate(str(i), pos, fontsize=8, ha="center")
        
        for i, p in enumerate(qubit_pos):
            pos=(a*p[0], a*p[1])
            fc2='darkgrey'
            if i in list(actions_k[:,1]):
                fc2 = 'red'


            circle2 = plt.Circle( pos , radius=a*0.25, ec='k', fc=fc2)
            ax2.add_patch(circle2)
            ax2.annotate(str(i), pos, fontsize=8, ha="center")

        for i, p in enumerate(qubit_pos):
            pos=(a*p[0], a*p[1])
            fc3='darkgrey'
            if p in list(initial_flips_k)[0]:
                fc3 = 'magenta'


            circle3 = plt.Circle( pos , radius=a*0.25, ec='k', fc=fc3)
            ax3.add_patch(circle3)
            ax3.annotate(str(i), pos, fontsize=8, ha="center")

        ax1.set_xlim([-.1,1.1])
        ax1.set_ylim([-.1,1.1])
        ax1.set_aspect(1)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title("actions agent")
        ax2.set_xlim([-.1,1.1])
        ax2.set_ylim([-.1,1.1])
        ax2.set_aspect(1)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_title("actions MWPM")
        ax1.axis('off')
        ax2.axis('off')
        ax3.set_xlim([-.1,1.1])
        ax3.set_ylim([-.1,1.1])
        ax3.set_aspect(1)
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_title("initial qubit flips")
        ax3.axis('off')
        plt.show()



def plot_single_box_dynamic(path_plot, rewards_agent):

    plt.figure(figsize=(7,6))

    plt.boxplot(rewards_agent)
    plt.xlabel("Agent", fontsize=16)
    plt.ylabel("Reward", fontsize=16)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.savefig(path_plot)
    plt.show()

