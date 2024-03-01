from stable_baselines3 import DQN
from environments.toric_game_static_env import ToricGameEnv, ToricGameEnvFixedErrs, ToricGameEnvLocalErrs
import os
from agents.custom_callback import SaveOnBestTrainingRewardCallback
from stable_baselines3.common.monitor import Monitor
from functions.plot_functions import  plot_log_results 




class DQN_agent:
    def __init__(self, initialisation_settings, log):

        '''
        This class initialises, trains, or loads a trained DQN model using the provided settings.

        Args:
            initialisation_settings (dict): the settings the DQN model should be initialised on.
            log (boolean): boolean specifying whether the training progress should be logged or not.

        self.storing_folder (str): specify the folder the model can be saved to/ accessed trough, and where the log files should be accessed from.
        
        
        '''

        self.storing_folder = "/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/MRP_lindsay"

        self.initialisation_settings=initialisation_settings
        # Create log dir
        self.log=log
        if self.log:
            self.log_dir = f"{self.storing_folder}/log_dirs/log_dir_dqn"
            os.makedirs(self.log_dir, exist_ok=True)



        # INITIALISE MODEL
        self.initialise_model()

    def initialise_model(self):
        # INITIALISE ENVIRONMENT
        print("initialising the environment and model...")
        if self.initialisation_settings['fixed']:
            self.env = ToricGameEnvFixedErrs(self.initialisation_settings)
        else:
            if not self.initialisation_settings['correlated']:
                self.env = ToricGameEnv(self.initialisation_settings)
            else:
                self.env = ToricGameEnvLocalErrs(self.initialisation_settings)


        if self.log:
            self.env = Monitor(self.env, self.log_dir)
            self.callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=self.log_dir)
            
        # INITIALISE MODEL
        dqn= DQN
        policy = "MlpPolicy"
        
        self.model = dqn(policy, self.env,learning_rate=self.initialisation_settings['lr'], verbose=0,exploration_fraction=self.initialisation_settings['exp_frac'], exploration_initial_eps=self.initialisation_settings['exp_init'], exploration_final_eps=self.initialisation_settings['exp_fin'], buffer_size=self.initialisation_settings['buff']) 

        print("initialisation done")
        print(self.model.policy)

    def change_environment_settings(self, settings):
        '''
        Changes the environment to new settings.

        Args:
            settings (dict): the settings the environment should be changed to.

        '''

        print("changing environment settings...")
        if settings['fixed']:
            self.env = ToricGameEnvFixedErrs(settings)
        else:
            if not self.initialisation_settings['correlated']:
                self.env = ToricGameEnv(settings)
            else:
                self.env = ToricGameEnvLocalErrs(settings)


        if self.log:
            self.env = Monitor(self.env, self.log_dir, override_existing=False)
            self.callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=self.log_dir)
        
        self.model.set_env(self.env)

        print("changing settings done")

    def train_model(self, save_model_path):
        print("training the model...")
        if self.log:
            self.model.learn(total_timesteps=self.initialisation_settings['total_timesteps'], progress_bar=True, callback=self.callback)
            plot_log_results(self.log_dir, save_model_path)
        else:
            self.model.learn(total_timesteps=self.initialisation_settings['total_timesteps'], progress_bar=True)
    
        self.model.save(f"trained_models/dqn_{save_model_path}")
        print("training done")

    def load_model(self, load_model_path):
        print("loading the model...")

        self.model=DQN.load(f"trained_models/dqn_{load_model_path}")
        print("loading done")