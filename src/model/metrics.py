from collections import OrderedDict
import sys
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

sys.path.append('.')

#from src.features.dtw_wd import dtw_windowed

def linear_relation(M, O):
    '''Compute linear relation M_i = offset + slope * O_i
    Input:
        M: Model prediction, shape (batches, time_forward)
        O: observational data
    Output:
        offset
        slope
        sigma_o: standard deviation offset
        sigma_s: standard deviation slope
    '''
    num_examples = M.shape[0]
    delta = num_examples*np.sum(O**2, axis=0) - np.sum(O, axis=0)**2

    offset, slope = [], []
    for i in range(M.shape[1]):
        res = np.polynomial.Polynomial.fit(O[:, i], M[:, i], deg=1)
        offset.append(res.convert().coef[0])
        slope.append(res.convert().coef[1])

    sigma_m = np.std(M, axis=0)

    sigma_o = sigma_m * np.sqrt(np.sum(O**2, axis=0) / delta)
    sigma_s = sigma_m * np.sqrt(num_examples/delta)

    return np.array(offset), np.array(slope), sigma_o, sigma_s


def pearson_correlation(M, O):
    corr = []
    for i in range(M.shape[1]):
        corr.append(np.corrcoef(M[:, i], O[:, i])[0, 1])
    return np.array(corr)


def rmse(M, O):
    return np.sqrt(mean_squared_error(O, M, multioutput='raw_values'))


def mae(M, O):
    return mean_absolute_error(O, M, multioutput='raw_values')


def mean_error(M, O):
    return np.sum(M - O, axis=0) / M.shape[0]


def prediction_efficiency(M, O):
    return 1 - np.sum((M - O)**2, axis=0) / np.sum((O - np.mean(O, axis=0))**2, axis=0)


def evaluate(M, O):
    A, B, sA, sB = linear_relation(M, O)
    R = pearson_correlation(M, O)
    rmse_ = rmse(M, O)
    mae_ = mae(M, O)
    me_ = mean_error(M, O)
    pe_ = prediction_efficiency(M, O)
    return {'A': A,
            'B': B,
            'sigmaA': sA,
            'sigmaB': sB,
            'R': R,
            'RMSE': rmse_,
            'MAE': mae_,
            'ME': me_,
            'PE': pe_
            }


def print_evaluation(ev):
    print('##################################################')
    for k in ev:
        print("{:>7}: {}".format(k, np.array2string(ev[k], precision=3)))
    print('##################################################')

    #evaluate_active_states(pred[:,0], truth[:,0], np.arange(20, -140, -10))

    #print('##################################################')
    #latency_metric(truth[:,0], pred[:,0], 6)
    #print('##################################################')


def store_to_csv(result, filename, append=0):
    mode = 'w'
    if append:
        mode = 'a'
    num_param = len(list(result.keys()))
    size = len(result[list(result.keys())[0]])
    with open(filename, mode) as f:
        if not append: # Write file headers
            ind = 1
            f.write("time,")
            for key in result:
                f.write('{}'.format(key))
                if ind < num_param:
                    f.write(',')
                    ind += 1
            f.write('\n')
        for t in range(size):
            ind = 1
            f.write('t+{},'.format(t+1+append))
            for key in result:
                f.write('{0:.3f}'.format(result[key][t]))
                if ind < num_param:
                    f.write(',')
                    ind += 1
            f.write('\n')


def apply_threshold(M, O, threshold):

    tp_, fp_, tn_, fn_ = np.zeros((4, M.shape[1]))

    for i in range(M.shape[1]):
        pred = M[:, i] < threshold
        obs = O[:, i] < threshold
        for j in range(pred.shape[0]):
            if pred[j] and obs[j]:
                tp_[i] += 1
            elif ~pred[j] and ~obs[j]:
                tn_[i] += 1
            elif ~pred[j]:
                fn_[i] += 1
            else:
                fp_[i] += 1


    return tp_, fp_, tn_, fn_


def heidke_measure(t_pos, f_pos, t_neg, f_neg):
    num = 2*(t_pos*t_neg - f_pos*f_neg)
    den = (t_pos+f_neg)*(f_neg + t_neg) + (t_pos + f_pos)*(f_pos + t_neg)
    return num/den

def pod(t_pos, f_neg):
    return t_pos / (t_pos + f_neg)

def pofd(f_pos, t_neg):
    return f_pos / (f_pos + t_neg)

def far(f_pos, t_pos):
    return f_pos / (f_pos + t_pos)

def freq_bias(t_pos, f_pos, f_neg):
    return (t_pos + f_pos) / (t_pos + f_neg)

def eval_active_state(M, O, threshold):
    tp_, fp_, tn_, fn_ = apply_threshold(M, O, threshold)
    H = heidke_measure(tp_, fp_, tn_, fn_)
    pod_ = pod(tp_, fn_)
    pofd_ = pofd(fp_, tn_)
    far_ = far(fp_, tp_)
    fb_ = freq_bias(tp_, fp_, fn_)

    return {'Heidke': H,
            'pod': pod_,
            'pofd': pofd_,
            'far': far_,
            'fb': fb_}


def plot_roc(M, O, thresholds):
    vals = np.zeros((len(thresholds), 2, M.shape[1]))
    for i, t in enumerate(thresholds):
        tp_, fp_, tn_, fn_ = apply_threshold(M, O, t)
        pod_ = pod(tp_, fn_)
        pfd_ = pofd(fp_, tn_)
        vals[i] = np.array([pfd_, pod_])

    for k in range(M.shape[1]):
        plt.scatter(vals[:, 0, k], vals[:, 1, k])
        plt.scatter(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01), c='k', s=0.15)
        plt.title('ROC curve for t+{}'.format(k+1))
        plt.xlabel('POFD')
        plt.ylabel('POD')
        #plt.xlim(-0.1, 1.05)
        #plt.ylim(-0.1, 1.05)
        plt.show()


def evaluate_active_states(M, O, thresholds):
    for t_ in thresholds:
        res = eval_active_state(M, O, t_)

        print('Threshold: {}'.format(t_))
        for k in res:
            print("Key {:>6}: {}".format(k, np.array2string(res[k], precision=2)))

    plot_roc(M, O, thresholds)


def latency_metric(truth, pred, max_lat):
    r'''Checks for latency in the prediction
    Input:
        truth: Ordered timeseries of shape (timesteps, time_forward). Numpy array
        pred: Ordered timeseries of shape (timesteps, time_forward). Numpy array
    Output:
        Best fitting latency for each variable (time_forward, latency)
    '''
    bincounts = np.zeros((truth.shape[1], max_lat+1))
    for i in range(truth.shape[1]):
        path, _ = dtw_windowed(pred[:, i], truth[:, i], max_lat, True)
        bins, counts = np.unique(abs(path[0, :] - path[1, :]), return_counts=True)
        bincounts[i, bins] = counts
    
    return bins, bincounts
