'''
@author: Brecht Laperre
@description: dtw-measure algorithm optimized for very large time series (N > 1000)
'''

import numpy as np
from scipy.sparse import dia_matrix

def _eff_compute_dist_windowed(T, P, window):
    '''Compute squared distance between each point
    Optimized for large T and P with small windowsize
    '''
    assert window < len(T) # number of rows
    assert window < len(P) # number of columns
    if len(T) > len(P):
        rows = T
        cols = P
        switched = False
    else:
        rows = P
        cols = T
        switched = True

    delta_r = len(rows) - len(cols)
    D = np.ones((window*2 + 3, len(cols)))*np.infty

    for w in range(1, window+1): #1, 2, ..., window
        abov_diag = np.sqrt(np.square(rows[w:len(cols)+w] - cols[:len(rows)-w]))
        D[window+1-w, -len(abov_diag):] = abov_diag

        under_diag = np.sqrt(np.square(cols[w:] - rows[:-w-delta_r]))
        D[window+1+w, :len(under_diag)] = under_diag
    
    D[window+1] = np.sqrt(np.square(rows[:len(cols)]-cols))
    offset = np.array(np.arange(window+1, -window-2, -1))
    
    D = dia_matrix((D, offset), shape=(len(T), len(P)))

    if not switched:
        D = D.transpose()
    return D.tocsr()


def _eff_compute_cost_windowed(D, window):

    for i in range(1, window+1):
        D[0, i] += D[0, i-1]
        D[i, 0] += D[i-1, 0]
    

    for i in range(1, D.shape[0]):
        if i % 10000 == 0:
            print(i)
        if i <= window:
            for j in range(1, window+i+1):
                if D[i,j] == np.infty:
                    break
                D[i,j] += np.min((D[i-1, j-1], D[i-1, j], D[i, j-1]))
            #D[i,i+window] += np.min((D[i-1, i+window-1], D[i, i+window-1]))
        elif i + window < D.shape[1]:
            #D[i,i-window] += np.min((D[i-1, i-window-1], D[i-1, i-window]))
            for w in range(-window, window+1):
                if D[i,i+w] == np.infty:
                    break
                D[i,i+w] += np.min((D[i-1, i+w-1], D[i-1, i+w], D[i, i+w-1]))
            #D[i,i+window] += np.min((D[i-1, i+window-1], D[i, i+window-1]))
        else:
            #D[i,i-window] += np.min((D[i-1, i-window-1], D[i-1, i-window]))
            for j in range(i-window, D.shape[1]):
                if D[i,j] == np.infty:
                    break
                D[i,j] += np.min((D[i-1, j], D[i-1, j-1], D[i, j-1]))
    return D


def _eff_find_path(D):

    path = np.zeros((2, D.shape[0] + D.shape[1]), dtype=int)

    loc_i, loc_j = D.shape[0]-1, D.shape[1]-1
    ind = D.shape[0] + D.shape[1] - 1
    path[:, ind] = np.array([loc_i, loc_j])
    while (loc_i != 0) and (loc_j != 0):
        v = np.array([D[loc_i-1, loc_j-1], D[loc_i-1, loc_j], D[loc_i, loc_j-1]])
        _min = np.argmin(v)
        if _min == 0 or _min == 1:
            loc_i -= 1
        if _min == 0 or _min == 2:
            loc_j -= 1
        ind -= 1
        path[:, ind] = np.array([loc_i, loc_j])

    if loc_j != 0:
        for k in range(loc_j-1, 0, -1):
            ind -= 1
            path[:, ind] = np.array([loc_i, k])
        ind -= 1
    if loc_i != 0:
        for k in range(loc_i-1, 0, -1):
            ind -= 1
            path[:, ind] = np.array([k, loc_j])
        ind -= 1

    return D[-1, -1], path[:, ind:]


def eff_dtw_measure(T, P, w, asymmetric=False):
    '''Modified DTW for large matrices. Uses sparse matrix representation for calculations
    Input:
        T: True timeseries
        P: Prediction
        w: windowsize
    Output:
        D: matrix of size T x P containing the cost of the mapping
        path: least cost path mapping P to T
        cost: DTW cost of the path
    '''
    assert len(T)
    assert len(P)

    D = _eff_compute_dist_windowed(T, P, w)
    if asymmetric:
        rows, cols = D.nonzero()
        for row,col in zip(rows,cols):
            if row - col < 0:
                D[row, col] = np.infty
    D = _eff_compute_cost_windowed(D, w)
    cost, path = _eff_find_path(D)

    return D, path, cost


if __name__ == '__main__':

    QQ = np.array([11, 17, 13, 8, 10, 5, 4, 5, 6])
    SS = np.array([12, 15, 18, 12, 7, 5, 2, 3, 1])
    PATH, COST = eff_dtw_measure(SS, QQ, 2, True)
    print(PATH)
    print(COST)
