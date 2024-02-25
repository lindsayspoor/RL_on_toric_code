from __future__ import division
from typing import Any
import numpy as np

import gymnasium as gym
#import gym
from gymnasium.utils import seeding
from gymnasium import spaces
import matplotlib.pyplot as plt




### Environment
class ToricGameEnv(gym.Env):
    '''
    ToricGameEnv environment. Effective single player game.
    '''

    def __init__(self, settings):
        """
        Args:
            opponent: Fixed
            board_size: board_size of the board to use
        """

        self.settings=settings

        self.board_size = settings['board_size']
        self.channels = [0]
        self.memory = False
        self.error_rate = settings['error_rate']
        self.logical_error_reward=settings['l_reward']
        self.continue_reward=settings['c_reward']
        self.success_reward=settings['s_reward']
        self.mask_actions=settings['mask_actions']
        self.illegal_action_reward = settings['i_reward']
        self.N = settings['N']
        self.pauli_opt=0
        self.max_steps=settings['max_moves']
        self.counter=0
        self.lambda_value=1

        # Keep track of the moves
        self.qubits_flips = [[],[]]
        self.initial_qubits_flips = [[],[]]


        # Empty State
        self.state = Board(self.board_size)
        self.done = None
        self.logical_error = None


        self.parity_check_matrix_plaqs = self.construct_parity_check_plaqs()
        self.dist_matrix = self.construct_distance_matrix()

        self.observation_space = spaces.MultiBinary(self.board_size*self.board_size) #3x3 plaquettes on which we can view syndromes
        self.action_space = spaces.discrete.Discrete(len(self.state.qubit_pos)) #0-17 qubits on which a bit-flip error can be introduced


    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**32
        return [seed1, seed2]
    

    def construct_distance_matrix(self):

        pair_distances = np.empty((len(self.state.qubit_pos), len(self.state.qubit_pos)))
        for i, p1 in enumerate(self.state.qubit_pos):
            for j, p2 in enumerate(self.state.qubit_pos):
                dist_nd_sq = 0
                for d in range(2):
                    dist_1d = abs(p2[d] - p1[d])
                    if dist_1d > ((self.board_size*2) / 2):  # check if d_ij > box_size / 2
                        dist_1d = (self.board_size*2) - dist_1d
                    dist_nd_sq += dist_1d ** 2
                pair_distances[i, j] = dist_nd_sq
        pair_distances = np.sqrt(pair_distances)

        return pair_distances
    

    def construct_weighted_matrix(self,dist_matrix, q):

        selected_dist_matrix = dist_matrix[q]/(2*self.board_size)
        weighted_matrix = (1-selected_dist_matrix)
        where_zero = np.argwhere(weighted_matrix == np.max(weighted_matrix))[0]
        weighted_matrix[where_zero] = 0

        return weighted_matrix




    def construct_parity_check_plaqs(self):

        #construct partiy check matrix for plaquette positions w.r.t qubit positions (needed for MWPM decoding)
        parity_check_matrix_plaqs = np.zeros((len(self.state.plaquet_pos), len(self.state.qubit_pos)))
        
        for plaq_ind, plaq_pos in enumerate(self.state.plaquet_pos):
            neighbours = self.find_neighboring_qubits(plaq_pos)
            for neighbour in neighbours:
                parity_check_matrix_plaqs[plaq_ind][neighbour] = 1 

        return parity_check_matrix_plaqs      


    def find_neighboring_qubits(self, plaq):
        '''Find qubits adjacent to given plaquette.'''

        neighboring_qubits=[]
        a,b = plaq[0], plaq[1]
        neighboring_qubit_pos = [[(a-1)%(2*self.state.size),b%(2*self.state.size)],[a%(2*self.state.size),(b-1)%(2*self.state.size)],[a%(2*self.state.size),(b+1)%(2*self.state.size)],[(a+1)%(2*self.state.size),b%(2*self.state.size)]]
        for i in neighboring_qubit_pos:
            neighboring_qubits.append(self.state.qubit_pos.index(i))

        return neighboring_qubits

    def action_masks(self):

        self.action_masks_list=np.zeros((len(self.state.qubit_pos)))
        self.action_masks_list[:]=False
        for i in self.state.syndrome_pos:
            mask_pos = self.find_neighboring_qubits(i)
            self.action_masks_list[mask_pos]=True
                


        self.action_masks_list=list(self.action_masks_list)

        return self.action_masks_list


    def check_correction(self,grid_q):
        """(tested for random ones):Check if the correction is correct(no logical X gates)
        input:
            grid_q: grid of qubit with errors and corrections
        output:
            corrected: boolean whether correction is correct.
        """
        # correct if even times logical X1,X2=> even number of times through certain edges
        # upper row = X1
        if sum(grid_q[0]) % 2 == 1:
            return (False, 'X1')
        # odd rows = X2
        if sum([grid_q[x][0] for x in range(1, len(grid_q), 2)]) == 1:
            return (False, 'X2')

        # and if all stabilizers give outcome +1 => even number of qubit flips for each stabilizer
        # is this needed? or assume given stabilizer outcome is corrected for sure?
        for row_idx in range(int(len(grid_q) / 2)):
            for col_idx in range(len(grid_q[0])):
                all_errors = 0
                all_errors += grid_q[2 * row_idx][col_idx]  # above stabilizer
                all_errors += grid_q[2 * row_idx + 1][col_idx]  # left of stabilizer
                if row_idx < int(len(grid_q) / 2) - 1:  # not the last row
                    all_errors += grid_q[2 * (row_idx + 1)][col_idx]
                else:  # last row
                    all_errors += grid_q[0][col_idx]
                if col_idx < len(grid_q[2 * row_idx + 1]) - 1:  # not the last column
                    all_errors += grid_q[2 * row_idx + 1][col_idx + 1]
                else:  # last column
                    all_errors += grid_q[2 * row_idx + 1][0]
                if all_errors % 2 == 1:
                    return (False, 'stab', row_idx, col_idx)  # stabilizer gives error -1

        return (True, 'end')


    def check(self, flips, error):


        grid_q = [[0 for col in range(self.board_size)] for row in range(2 * self.board_size)]
        grid_q=np.array(grid_q)
        for i in flips[0]:
            flip_index = [j==i for j in self.state.qubit_pos]
            flip_index = np.reshape(flip_index, newshape=(2*self.board_size, self.board_size))
            flip_index = np.argwhere(flip_index)
            grid_q[flip_index[0][0],flip_index[0][1]]+=1 % 2
        grid_q = list(grid_q)

        correction_error = self.check_correction(grid_q)[0]
        if (correction_error == error):
            print("oeps,mwpm", flips, error)


        logical_error = self.check_logical_error()
        if not (logical_error==error):
            print("oeps, agent", flips, error)
        




    def generate_errors(self, allow_empty=False):

        self.state.reset()

        # Let the opponent do it's initial evil
        self.qubits_flips = [[],[]]
        self.initial_qubits_flips = [[],[]]

        self._set_initial_errors()

        self.logical_error=self.check_logical_error()
        self.done=self.state.has_no_syndromes()


        #for evaluation
        if not allow_empty and (self.done or self.logical_error):
            self.generate_errors()

        return self.state.encode(self.channels, self.memory)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None, allow_empty=False) -> tuple[Any, dict[str, Any]]:
         super().reset(seed=seed, options=options)
        
         self.counter=0
         initial_observation = self.generate_errors(allow_empty)

         return initial_observation, {'state': self.state, 'message':"reset"}


    def close(self):
        self.state = None

    def render(self, mode="human", close=False):
        fig, ax = plt.subplots()
        a=1/(2*self.board_size)

        for i, p in enumerate(self.state.plaquet_pos):
            if self.state.op_values[0][i]==1:
                fc='darkorange'
                plaq = plt.Polygon([[a*p[0], a*(p[1]-1)], [a*(p[0]+1), a*(p[1])], [a*p[0], a*(p[1]+1)], [a*(p[0]-1), a*p[1]] ], fc=fc)
                ax.add_patch(plaq)

        for i, p in enumerate(self.state.star_pos):
            if self.state.op_values[1][i]==1:
                fc = 'green'
                plaq = plt.Polygon([[a*p[0], a*(p[1]-1)], [a*(p[0]+1), a*(p[1])], [a*p[0], a*(p[1]+1)], [a*(p[0]-1), a*p[1]] ], fc=fc)
                ax.add_patch(plaq)

        # Draw lattice
        for x in range(self.board_size):
            for y in range(self.board_size):
                pos=(2*a*x, 2*a*y)
                width=a*2
                lattice = plt.Rectangle( pos, width, width, fc='none', ec='black' )
                ax.add_patch(lattice)

        for i, p in enumerate(self.state.qubit_pos):
            pos=(a*p[0], a*p[1])
            fc='darkgrey'
            if self.state.qubit_values[0][i] == 1 and self.state.qubit_values[1][i] == 0:
                fc='darkblue'
            elif self.state.qubit_values[0][i] == 0 and self.state.qubit_values[1][i] == 1:
                fc='darkred'
            elif self.state.qubit_values[0][i] == 1 and self.state.qubit_values[1][i] == 1:
                fc='darkmagenta'
            circle = plt.Circle( pos , radius=a*0.25, ec='k', fc=fc)
            ax.add_patch(circle)
            plt.annotate(str(i), pos, fontsize=8, ha="center")

        ax.set_xlim([-.1,1.1])
        ax.set_ylim([-.1,1.1])
        ax.set_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.axis('off')
        plt.show()

    def step(self, location):

        '''
        Args:
            location: coord of the qubit to flip
            operator: Pauli matrix to apply
        Return:
            observation: board encoding,
            reward: reward of the game,
            done: boolean,
            info: state dict
        '''

        if self.counter==self.max_steps:
            return self.state.encode(self.channels, self.memory), self.continue_reward, self.done, True,{'state': self.state, 'message':"max steps reached"}


        if self.done:
            if self.logical_error:
                return self.state.encode(self.channels, self.memory), self.logical_error_reward, self.done, False,{'state': self.state, 'message':"logical_error"}
            else:
                return self.state.encode(self.channels, self.memory), self.success_reward, self.done, False,{'state': self.state, 'message':"success"}

   
        self.qubits_flips[0].append(location)
   
        self.state.act(self.state.qubit_pos[location], self.pauli_opt)

        self.counter+=1
                
        if self.state.has_no_syndromes()==False:
            self.done = False
            #self.logical_error = self.check_logical_error()
            if self.mask_actions==False:

                self.action_masks_list = self.action_masks()
                if self.action_masks_list[location] == False:
                    return self.state.encode(self.channels, self.memory), self.illegal_action_reward, self.done, False,{'state': self.state, 'message':"illegal action, continue"}
                else:
                    return self.state.encode(self.channels, self.memory), self.continue_reward, self.done, False,{'state': self.state, 'message':"continue"}
            
            else:
                return self.state.encode(self.channels, self.memory), self.continue_reward, self.done, False,{'state': self.state, 'message':"continue"}
        
        else:
            self.done=True
            self.logical_error = self.check_logical_error()

            if self.logical_error:
                return self.state.encode(self.channels, self.memory), self.logical_error_reward, self.done, False,{'state': self.state, 'message':"logical_error"}
            else:
                return self.state.encode(self.channels, self.memory), self.success_reward, self.done, False,{'state': self.state, 'message':"success"}



    def _set_initial_errors(self):
        ''' Set random initial errors with an %error_rate rate
            but report only the syndrome
        '''
        # Probabilitic mode
        # Pick random sites according to error rate

        for q in self.state.qubit_pos:    
            if np.random.rand() < self.error_rate:
                self.initial_qubits_flips[0].append( q )
                self.state.act(q, self.pauli_opt)


       
        # Now unflip the qubits, they're a secret
        self.state.qubit_values = np.zeros((2, 2*self.board_size*self.board_size))





    def find_string(self,q,l_list, checked_plaqs):
        '''Finds the error strings on the board and checks whether they form closed loops or not.
        returns: 
        - l_list: list containing the qubits composing each error string
        - closed: boolean, True of the loop from l_list forms a closed loop
        '''

        closed = False
        coord = self.state.qubit_pos[q]

        neighboring_plaqs = self.state.adjacent_plaquettes(coord)
        l_list.append(q)

        for i in neighboring_plaqs:
            if i in self.state.syndrome_pos:
                return l_list, closed, checked_plaqs
        
        
        next_plaq = neighboring_plaqs[0]

        if next_plaq in checked_plaqs:
            next_plaq=neighboring_plaqs[1]

            if next_plaq in checked_plaqs:
                return l_list, closed, checked_plaqs #if a plaquette is already checked on all its neighboring qubits it doesn't need to be checked again
        

        checked_plaqs.append(next_plaq)


        neighboring_qubits = self.find_neighboring_qubits(next_plaq)
        neighboring_qubits.remove(q)


        for i in neighboring_qubits:
            if i ==l_list[0]:
                closed = True #closed loop
                return l_list, closed, checked_plaqs #closed loop
                
            
            if self.state.hidden_state_qubit_values[0][i]==1: #is this neighboring qubit a flipped one?

                l_list, closed, checked_plaqs = self.find_string(i, l_list, checked_plaqs) #check again for next flipped qubit if it ends at a syndrome point or not.



        return l_list, closed, checked_plaqs 

    def number_of_times_boundary(self, l_list):
        '''counts how many times an error string crosses the a boundary.'''

        nx = 0
        ny = 0
        for i in l_list:
            if i in self.state.left_boundary_qubits:
                ny+=self.state.hidden_state_qubit_values[0][i]
            if i in self.state.bottom_boundary_qubits:
                nx+=self.state.hidden_state_qubit_values[0][i]

        return nx,ny


    def flatten(self, xss):
        return [x for xs in xss for x in xs]
    
    def check_logical_error(self):
        '''Checks if an error string has a closed loop and if so it checks if the loop is non-trivial, 
        resulting in a logical error.
        return: False -> no logical error, True -> logical error.
        '''


        Nx = 0 
        Ny = 0 #counts the number of logical errors on the board. If Nx or Ny are an even number, this means that 2 logical errors canceled each other.

        l_list_global=[]
        for q in self.state.boundary_qubits:
            flat_l_list_global = self.flatten(l_list_global)

            if q in flat_l_list_global:
                continue

            if self.state.hidden_state_qubit_values[0][q]==1: #only check for the flipped qubits on the boundary of the board
                l_list = []
                checked_plaqs=[]
                l_list, closed, checked_plaqs = self.find_string(q, l_list, checked_plaqs)

                if closed:
                    l_list_global.append(l_list)
                    nx,ny = self.number_of_times_boundary(l_list)
                    Nx+=nx
                    Ny+=ny

        if (Nx%2==1) or (Ny%2==1):
            return True #logical error, non-trivial loop

        return False #no logical error
        

            


class ToricGameEnvFixedErrs(ToricGameEnv):
    def __init__(self, settings):
        super().__init__(settings)
        

    def _set_initial_errors(self):
        ''' Set random initial errors with an %error_rate rate
            but report only the syndrome
        '''

        #error = True
        for q in np.random.choice(len(self.state.qubit_pos), self.N, replace=False):
        #for q in [3,9,15,16,0]:
            q = self.state.qubit_pos[q]
            self.initial_qubits_flips[0].append(q)
            self.state.act(q, self.pauli_opt)

        # Now unflip the qubits, they're a secret
        self.state.qubit_values = np.zeros((2, 2*self.board_size*self.board_size))


        #self.check(self.initial_qubits_flips, error)

class ToricGameEnvLocalErrs(ToricGameEnv):
    def __init__(self, settings):
        super().__init__(settings)
        

    def _set_initial_errors(self):
        ''' Set random initial errors with an %error_rate rate and flip next qubits with a correlation to the former flipped ones
            but report only the syndrome
        '''



        probabilities = np.ones((len(self.state.qubit_pos)))

        #
        number_flips = np.random.binomial(len(self.state.qubit_pos),self.error_rate) #choose the amount of qubits to flip according to error rate

        for i in range(number_flips):
            probabilities=probabilities/np.sum(probabilities)
            q = np.random.choice(len(self.state.qubit_pos), p=probabilities) #choose which qubit to flip according to 'probabilities', uniformly distributed at first flip

            self.initial_qubits_flips[0].append( self.state.qubit_pos[q] )
            self.state.act(self.state.qubit_pos[q], self.pauli_opt)

            weighted_matrix = self.construct_weighted_matrix(self.dist_matrix, q)

            probabilities = probabilities*weighted_matrix #update 'probabilities' such that the qubits close to the former qubit flips have a higher chance to flip as well


        # Now unflip the qubits, they're a secret
        self.state.qubit_values = np.zeros((2, 2*self.board_size*self.board_size))




class Board(object):
    '''
    Basic Implementation of a ToricGame Board, actions are int [0,2*board_size**2)
    o : qubit
    P : plaquette operator
    x : star operator

    x--o---x---o---x---o---
    |      |       |
    o  P   o   P   o   P
    |      |       |
    x--o---x---o---x---o---
    |      |       |
    o  P   o   P   o   P
    |      |       |
    x--o---x---o---x---o---
    |      |       |
    o  P   o   P   o   P
    |      |       |

    '''
    @staticmethod
    def component_positions(size):
        qubit_pos   = [[x,y] for x in range(2*size) for y in range((x+1)%2, 2*size, 2)]
        plaquet_pos = [[x,y] for x in range(1,2*size,2) for y in range(1,2*size,2)]
        star_pos    = [[x,y] for x in range(0,2*size,2) for y in range(0,2*size,2)]


        return qubit_pos, plaquet_pos, star_pos


    def __init__(self, board_size):
        self.size = board_size

        # Real-space locations
        self.qubit_pos, self.plaquet_pos, self.star_pos  = self.component_positions(self.size)

        # Define here the logical error for efficiency
        self.left_boundary_positions, self.bottom_boundary_positions, self.left_boundary_qubits, self.bottom_boundary_qubits, self.boundary_qubits = self.define_boundary(self.qubit_pos)

        self.reset()

    def define_boundary(self, qubit_pos):

        left_boundary_positions = [[0,x] for x in range(1, 2*self.size, 2)]        
        bottom_boundary_positions = [[x,0] for x in range(1, 2*self.size, 2)]

        boundary_qubits = []
        left_boundary_qubits = []
        bottom_boundary_qubits = []
        for i in left_boundary_positions:
            boundary_qubit = qubit_pos.index(i)
            left_boundary_qubits.append(boundary_qubit)
            boundary_qubits.append(boundary_qubit)
        for i in bottom_boundary_positions:
            boundary_qubit = qubit_pos.index(i)
            bottom_boundary_qubits.append(boundary_qubit)
            boundary_qubits.append(boundary_qubit)
        
        return left_boundary_positions, bottom_boundary_positions, left_boundary_qubits, bottom_boundary_qubits, boundary_qubits


    def reset(self):
        
        self.qubit_values = np.zeros((2, 2*self.size*self.size))
        self.hidden_state_qubit_values = np.zeros((2, 2*self.size*self.size))  #make hidden state that contains the information about the initially flipped qubits (not visible to the agent)
        self.op_values = np.zeros((2, self.size*self.size))

        self.syndrome_pos = [] # Location of syndromes


    def adjacent_plaquettes(self, coord):
        '''Find plaquettes that the flipped qubit is a part of '''
        plaqs = []
        if coord[0] % 2 == 0:
            plaqs += [ [ (coord[0] + 1) % (2*self.size), coord[1] ], [ (coord[0] - 1) % (2*self.size), coord[1] ] ]
        else:
            plaqs += [ [ coord[0], (coord[1] + 1) % (2*self.size) ], [ coord[0], (coord[1] - 1) % (2*self.size) ] ]
        return plaqs




    def act(self, coord, operator):
        '''
            Args: input action in the form of position [x,y]
            coord: real-space location of the qubit to flip
        '''


        qubit_index=self.qubit_pos.index(coord)

        # Flip it!
        self.qubit_values[0][qubit_index] = (self.qubit_values[0][qubit_index] + 1) % 2
        self.hidden_state_qubit_values[0][qubit_index] = (self.hidden_state_qubit_values[0][qubit_index] + 1) % 2




        plaqs = self.adjacent_plaquettes(coord)

        # Update syndrome positions
        # TODO: Maybe update to boolean list?
        for plaq in plaqs:
            if plaq in self.syndrome_pos:
                self.syndrome_pos.remove(plaq)
            else:
                self.syndrome_pos.append(plaq)

            # The plaquette or vertex operators are only 0 or 1
            # TODO: This is always true?
            if plaq in self.plaquet_pos:
                op_index = self.plaquet_pos.index(plaq)
                channel = 0
        
            self.op_values[channel][op_index] = (self.op_values[channel][op_index] + 1) % 2





    def has_no_syndromes(self):

        # Are all syndromes removed?
        return len(self.syndrome_pos) == 0 #False if it has syndromes, True if there are no syndromes






    def has_odd_number_of_errors_on_boundary(self):

        # Check for Z logical error
        zerrors = [0,0]
        for pos in self.left_boundary_positions:
            qubit_index = self.qubit_pos.index(pos)
            zerrors[0] += self.qubit_values[ qubit_index ]

        for pos in self.bottom_boundary_positions:
            qubit_index = self.qubit_pos.index(pos)
            zerrors[1] += self.qubit_values[ qubit_index ]

        if (zerrors[0]%2 == 1) or (zerrors[1]%2 == 1):
            return True

        return False


    def __repr__(self):
        ''' representation of the board class
            print out board_state
        '''
        return f"Qubit Values: {self.qubit_values}, Operator values: {self.op_values}"

    def encode(self, channels, memory):
        '''Return: np array
            np.array(board_size, board_size): state observation of the board
        '''
        # In case of uncorrelated noise for instance, we don't need information
        # about the star operators

        img=np.array([])
        for channel in channels:
            img = np.concatenate((img, self.op_values[channel]))
            if memory:
                img = np.concatenate((img, self.qubit_values[channel]))

        return img

    def image_view(self, number=False, channel=0):
        image = np.empty((2*self.size, 2*self.size), dtype=object)
        #print(image)
        for i, plaq in enumerate(self.plaquet_pos):
            if self.op_values[0][i] == 1:
                image[plaq[0], plaq[1]] = "P"+str(i) if number else "P"
            elif self.op_values[0][i] == 0:
                image[plaq[0], plaq[1]] = "x"+str(i) if number else "x"
        for i,plaq in enumerate(self.star_pos):
            if self.op_values[1][i] == 1:
                image[plaq[0], plaq[1]] = "S"+str(i) if number else "S"
            elif self.op_values[1][i] == 0:
                image[plaq[0], plaq[1]] = "+"+str(i) if number else "+"

        for i,pos in enumerate(self.qubit_pos):
            image[pos[0], pos[1]] = str(int(self.qubit_values[channel,i]))+str(i) if number else str(int(self.qubit_values[channel, i]))

        return np.array(image)
