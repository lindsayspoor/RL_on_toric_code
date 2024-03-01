import numpy as np
from gymnasium import spaces

from environments.toric_game_static_env import ToricGameEnv, ToricGameEnvFixedErrs, Board


class IgnoreExtraActionBoard(Board):
    def __init__(self, board_size):
        super().__init__(board_size)
        self.ignored_counter = 0

    def act(self, coord, operator):
        if coord == len(self.qubit_pos):
            self.ignored_counter += 1
            print(f"ignored: {self.ignored_counter}")
            return
        return super().act(coord, operator)



class ToricGameEnvExtraAction(ToricGameEnv):
    def __init__(self, settings):
        super().__init__(settings)
        self.state = IgnoreExtraActionBoard(self.board_size)
        self.action_space = spaces.discrete.Discrete(len(self.state.qubit_pos)+1) # last action = 'do nothing'
                
    def action_masks(self):
        
        self.action_masks_list = np.zeros((len(self.state.qubit_pos)+1))
        self.action_masks_list[:] = False

        if self.state.has_no_syndromes() == True: # if there are no syndrome points on the board, the agent can only "do nothing"
            qubit_number = len(self.state.qubit_pos)
            self.action_masks_list[qubit_number] = True

        else:
            for i in self.state.syndrome_pos:
                mask_pos = self.find_neighboring_qubits(i)
                self.action_masks_list[mask_pos] = True
                


        self.action_masks_list=list(self.action_masks_list)

        return self.action_masks_list



class ToricGameEnvExtraActionFixed(ToricGameEnvFixedErrs):
    def __init__(self, settings):
        super().__init__(settings)
        self.state = IgnoreExtraActionBoard(self.board_size)
        self.action_space = spaces.discrete.Discrete(len(self.state.qubit_pos)+1) #last action = 'do nothing'
                
    def action_masks(self):
        
        self.action_masks_list = np.zeros((len(self.state.qubit_pos)+1))
        self.action_masks_list[:] = False

        if self.state.has_no_syndromes() == True: # if there are no syndrome points on the board, the agent can only "do nothing"
            qubit_number = len(self.state.qubit_pos)
            self.action_masks_list[qubit_number] = True

        else:
            for i in self.state.syndrome_pos:
                mask_pos = self.find_neighboring_qubits(i)
                self.action_masks_list[mask_pos] = True
                


        self.action_masks_list = list(self.action_masks_list)

        return self.action_masks_list