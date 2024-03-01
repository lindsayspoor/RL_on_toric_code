from __future__ import division
from typing import Any
import numpy as np
import gymnasium as gym
from gymnasium.utils import seeding
from gymnasium import spaces
import matplotlib.pyplot as plt



### Environment
class ToricGameEnvCNN(gym.Env):
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

        # Keep track of the moves
        self.qubits_flips = [[],[]]
        self.initial_qubits_flips = [[],[]]


        # Empty State
        self.state = Board(self.board_size)
        self.done = None
        self.logical_error = None


        self.observation_space = spaces.Box(low=0, high=1, shape=(1,self.board_size,self.board_size), dtype=np.uint8) # dxd plaquettes on which we can view syndromes
        self.action_space = spaces.discrete.Discrete(len(self.state.qubit_pos)) # 2dxd qubits on which a bit-flip error can be introduced


    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**32
        return [seed1, seed2]

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






    def generate_errors(self):

        self.state.reset()

        # Let the opponent do it's initial evil
        self.qubits_flips = [[],[]]
        self.initial_qubits_flips = [[],[]]

        self._set_initial_errors()

        self.logical_error=self.check_logical_error()
        self.done=self.state.has_no_syndromes()

        if self.done == True:
            self.generate_errors()


        return self.state.encode(self.channels, self.memory)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
         super().reset(seed=seed, options=options)

         initial_observation = self.generate_errors()

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

        if self.done:
            if self.logical_error:
                return self.state.encode(self.channels, self.memory), self.logical_error_reward, self.done, False,{'state': self.state, 'message':"logical_error"}
            else:
                return self.state.encode(self.channels, self.memory), self.success_reward, self.done, False,{'state': self.state, 'message':"success"}


   
        self.qubits_flips[0].append(location)
   
        self.state.act(self.state.qubit_pos[location], self.pauli_opt)


        if self.state.has_no_syndromes()==False:
            self.done = False
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
                closed = True # closed loop
                return l_list, closed, checked_plaqs # closed loop
                
            
            if self.state.hidden_state_qubit_values[0][i]==1: # is this neighboring qubit a flipped one?

                l_list, closed, checked_plaqs = self.find_string(i, l_list, checked_plaqs) # check again for next flipped qubit if it ends at a syndrome point or not.



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
        Ny = 0 # counts the number of logical errors on the board. If Nx or Ny are an even number, this means that 2 logical errors canceled each other.

        l_list_global=[]
        for q in self.state.boundary_qubits:
            flat_l_list_global = self.flatten(l_list_global)

            if q in flat_l_list_global:
                continue

            if self.state.hidden_state_qubit_values[0][q]==1: # only check for the flipped qubits on the boundary of the board
                l_list = []
                checked_plaqs=[]
                l_list, closed, checked_plaqs = self.find_string(q, l_list, checked_plaqs)

                if closed:
                    l_list_global.append(l_list)
                    nx,ny = self.number_of_times_boundary(l_list)
                    Nx+=nx
                    Ny+=ny

        if (Nx%2==1) or (Ny%2==1):
            return True # logical error, non-trivial loop

        return False # no logical error
        

            


class ToricGameEnvFixedErrsCNN(ToricGameEnvCNN):
    def __init__(self, settings):
        super().__init__(settings)
        

    def _set_initial_errors(self):
        ''' Set random initial errors with an %error_rate rate
            but report only the syndrome
        '''

        for q in np.random.choice(len(self.state.qubit_pos), self.N, replace=False):
            q = self.state.qubit_pos[q]
            self.initial_qubits_flips[0].append(q)
            self.state.act(q, self.pauli_opt)

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
        return len(self.syndrome_pos) == 0 # False if it has syndromes, True if there are no syndromes






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
            img = np.concatenate((img, self.op_values[channel])).reshape((1,self.size, self.size)).astype(np.uint8)
            if memory:
                img = np.concatenate((img, self.qubit_values[channel])).reshape((1,self.size, self.size)).astype(np.uint8)

        return img

    def image_view(self, number=False, channel=0):
        image = np.empty((2*self.size, 2*self.size), dtype=object)
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
