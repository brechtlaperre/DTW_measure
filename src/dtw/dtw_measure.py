'''
@author: Brecht Laperre
@description: dtw-measure algorithm
'''
import sys
import numpy as np

sys.path.append('.')

from src.dtw.dtw_m_e import eff_dtw_measure

def _compute_cost(D):

    D[0, :] = np.cumsum(D[0, :])
    D[:, 0] = np.cumsum(D[:, 0])

    for i in range(1, D.shape[0]):
        new_min = np.min(D[i-1:i+1, 0])
        for j in range(1, D.shape[1]):
            if D[i-1, j] < new_min:
                new_min = D[i-1, j]
                D[i, j] = D[i, j] + new_min
                continue
            else:
                D[i, j] = D[i, j] + new_min
                new_min = D[i-1, j] if D[i-1, j] < D[i, j] else D[i, j]

    return D

def _find_path(D):
    assert len(D[:, 0])
    assert len(D[0, :])

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


def _compute_dist(T, P):
    D = np.zeros((len(T), len(P)))

    for i, t in enumerate(T):
        D[i, i:] = np.sqrt((P[i:] - t) ** 2)

    if len(T) > len(P):
        D[len(P):, -1] = np.sqrt((T[len(P):] - P[-1]) ** 2)

    return D


def _compute_dist_windowed(T, P, window):
    assert window < len(P)
    assert len(P) == len(T)
    D = np.ones((len(T), len(P)))*np.infty

    for i, t in enumerate(T):
        #if i <= window:
        #    D[i, :i+1] = np.sqrt((P[:i+1] - t) ** 2)
        if len(P) - i <= window:
            D[i, i:] = np.sqrt((P[i:] - t) ** 2)
        else:
            D[i, i:i+window+1] = np.sqrt((P[i:i+window+1] - t) ** 2)

    if len(T) > len(P):
        D[len(P):, -1] = np.sqrt((T[len(P):] - P[-1]) ** 2)

    return D.T


def dtw_measure(T, P, w=None, asymmetric=False):
    '''Modified DTW, with constraint that prevents mapping predictions to truth values in the future
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

    if len(T) > 1000 or len(P) > 1000:
        D, path, cost = eff_dtw_measure(T, P, w, asymmetric)
        return path, cost


    if w is None:
        D = _compute_dist(P, T)
    else:
        D = _compute_dist_windowed(P, T, w)
    D = _compute_cost(D)

    cost, path = _find_path(D)

    return D, path, cost


if __name__ == '__main__':

    QQ = np.array([11, 17, 13, 8, 10, 5, 4, 5, 6])
    SS = np.array([12, 15, 18, 12, 7, 5, 2, 3, 1])
    DD, PATH, COST = dtw_measure(SS, QQ, w=2)
    print(DD)
    print('Path')
    print(PATH)
    print(COST)
