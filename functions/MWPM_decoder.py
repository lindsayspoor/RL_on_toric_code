import numpy as np
from pymatching import Matching



def check_correction(grid_q):
    """(tested for random ones):Check if the correction is correct(no logical X gates)
    input:
        grid_q: grid of qubit with errors and corrections
    output:
        corrected: boolean whether correction is correct.
    
    From: https://github.com/nanleij/The-toric-code
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
    # other way of checking: for each row, look if no errors on qubits, => no loop around torus,so no gate applied.
    # and similar for columns


def decode_MWPM_pymatching(parity_check_matrix,qubit_pos,obs0, initial_flips, evaluation_settings):


    matching  = Matching(parity_check_matrix)

    correction = matching.decode(obs0)

    grid_q = [[0 for col in range(evaluation_settings['board_size'])] for row in range(2 * evaluation_settings['board_size'])]
    grid_q=np.array(grid_q)

    #qubit_pos = agent.env.state.qubit_pos
    for i in initial_flips[0]:
        flip_index = [j==i for j in qubit_pos]
        flip_index = np.reshape(flip_index, newshape=(2*evaluation_settings['board_size'], evaluation_settings['board_size']))
        flip_index = np.argwhere(flip_index)

        grid_q[flip_index[0][0],flip_index[0][1]]+=1 % 2
    grid_q = list(grid_q)
    grid_q_initial=np.copy(grid_q)

    #qubit_pos = agent.env.state.qubit_pos
    correction_flips = np.reshape(correction, newshape=(2*evaluation_settings['board_size'], evaluation_settings['board_size']))


    MWPM_actions = np.argwhere(correction_flips.flatten()==1)
    residue_grid = (grid_q_initial + correction_flips) % 2

    
    check = check_correction(residue_grid)


    return check[0], MWPM_actions


          
