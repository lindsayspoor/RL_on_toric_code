import numpy as np
import networkx as nx
from pymatching import Matching







def matching_to_path(matchings, grid_q):

    """TESTED(for 1 matching):Add path of matchings to qubit grid
        input:
            matchings: array with tuples of two matched stabilizers as elements(stabilizer = tuple of coords)
            grid_q: grid of qubits with errors before correction
        output:
            grid_q: grid of qubits with all errors(correction=adding errors)
        """
    L = len(grid_q[0])
    for stab1, stab2 in matchings:
        error_path = [0, 0]
        row_dif = abs(stab1[0] - stab2[0])
        if row_dif > L - row_dif:
            # path through edge
            error_path[0] += 1
        col_dif = abs(stab1[1] - stab2[1])
        if col_dif > L - col_dif:
            # path through edge
            error_path[1] += 1
        last_row = stab1[0]
        if stab1[0] != stab2[0]:  # not the same row
            up_stab = min(stab1, stab2)
            down_stab = max(stab1, stab2)
            q_col = up_stab[1]  # column of the upper stabilizer
            last_row = down_stab[0]
            if error_path[0]:  # through edge
                for s_row in range(down_stab[0] - L, up_stab[0]):
                    q_row = (s_row + 1) * 2  # row under current stabilizer
                    grid_q[q_row][q_col] += 1  # make error = flip bit
            else:
                for s_row in range(up_stab[0], down_stab[0]):
                    q_row = (s_row + 1) * 2  # row under current stabilizer
                    grid_q[q_row][q_col] += 1

        if stab1[1] != stab2[1]:  # not the same col
            left_stab = min(stab1, stab2, key=lambda x: x[1])
            right_stab = max(stab1, stab2, key=lambda x: x[1])
            q_row = 2 * last_row + 1
            if error_path[1]:  # through edge
                for s_col in range(right_stab[1] - L, left_stab[1]):
                    q_col = s_col + 1  # col right of stabilizer
                    grid_q[q_row][q_col] += 1  # make error = flip bit
            else:
                for s_col in range(left_stab[1], right_stab[1]):
                    q_col = s_col + 1  # col right of stabilizer
                    grid_q[q_row][q_col] += 1
    return grid_q

def check_correction(grid_q):
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
    # other way of checking: for each row, look if no errors on qubits, => no loop around torus,so no gate applied.
    # and similar for columns

def decode_MWPM_method(qubit_pos,obs0_k, initial_flips, evaluation_settings):


    stab_errors = np.argwhere((obs0_k==1))


    path_lengths = []

    for stab1_idx in range(stab_errors.shape[0]-1):
        for stab2_idx in range(stab1_idx + 1, stab_errors.shape[0]):
            min_row_dif = min(abs(stab_errors[stab1_idx][0]-stab_errors[stab2_idx][0]), evaluation_settings['board_size']-abs(stab_errors[stab1_idx][0]-stab_errors[stab2_idx][0]))
            min_col_dif = min(abs(stab_errors[stab1_idx][1]-stab_errors[stab2_idx][1]), evaluation_settings['board_size']-abs(stab_errors[stab1_idx][1]-stab_errors[stab2_idx][1]))

            path_lengths.append([tuple(stab_errors[stab1_idx]),tuple(stab_errors[stab2_idx]), min_row_dif+min_col_dif])

    G = nx.Graph()

    for edge in path_lengths:
        G.add_edge(edge[0],edge[1], weight=-edge[2])

    matching = nx.algorithms.max_weight_matching(G, maxcardinality=True)

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


    matched_error_grid = matching_to_path(matching, grid_q)


    MWPM_grid = np.array(matched_error_grid)-grid_q_initial
    MWPM_actions = np.argwhere(MWPM_grid.flatten()==1)

    check = check_correction(matched_error_grid)



    return check[0], MWPM_actions





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


          
