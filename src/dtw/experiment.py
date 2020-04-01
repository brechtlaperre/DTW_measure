import pandas as pd
import numpy as np
import h5py
import sys

sys.path.append('.')

from src.dtw.dtw_measure import dtw_measure

def load_testing_sets(fname='data/processed/datasets.h5'):
    with h5py.File(fname, 'r') as f:
        test_in = f['test_sets/test_in'][:]
        test_out = f['test_sets/test_out'][:]
        predict = f['test_sets/prediction'][:]
        lookup = f['test_sets/lookup'][:]
    return test_in, test_out, predict, lookup.astype('datetime64[s]')

def extract_continuous_intervals(table):
    r'''Check lookup table for time discontinuities
    output: 
        Returns list of continouos times inside the lookup table
    '''
    lookup = pd.DataFrame(data=np.arange(table.shape[0]), index=pd.to_datetime(table[:,0]))
    lookup.index = pd.DatetimeIndex(lookup.index)
    # split = [g for n,g in lookup.groupby(pd.Grouper(freq='M')) if g.shape[0] != 0]

    min_size = 10
    timeseries = []
    
    #for month in split:
    missing = False
    series = lookup.index
    while len(series) > 0:
        # We can assume that the series starts from non-missing values, so the first diff gives sizes of continous intervals
        diff = pd.date_range(series[0], series[-1], freq='H').difference(series)
        if not missing:
            if len(diff) > 0:
                if pd.Timedelta(diff[0] - pd.Timedelta('1h') - series[0])/pd.Timedelta('1h') > min_size:
                    v1 = lookup.loc[series[0]][0]
                    v2 = lookup.loc[diff[0] - pd.Timedelta('1h')][0]
                    # print(series[0], diff[0] - pd.Timedelta('1h'))
                    timeseries.append([v1, v2])
                if pd.Timedelta(series[-1] - diff[-1] - pd.Timedelta('1h'))/pd.Timedelta('1h') > min_size:
                    v1 = lookup.loc[diff[-1] + pd.Timedelta('1h')][0]
                    v2 = lookup.loc[series[-1]][0]
                    # print(diff[-1] + pd.Timedelta('1h'), series[-1])
                    timeseries.append([v1, v2])
            else:
                # Only when diff is empty
                v1 = lookup.loc[series[0]][0]
                v2 = lookup.loc[series[-1]][0]
                # print(series[0], series[-1])
                timeseries.append([v1, v2])
        missing = not(missing)
        series = diff

    return np.array(timeseries)

def reformat_dtw_res(df, filename=None):
    '''Normalize the result from the dtw measure
    '''
    res = df.div(df.sum(axis=1), axis=0)

    shifts = np.array(['t+{}h'.format(i+1) for i in np.arange(res.shape[0])])
    res['Prediction'] = shifts.T
    res = res.set_index('Prediction')
    res.columns = ['{}h'.format(i) for i in res.columns]
    res = res.apply(lambda x: round(x, 3))
    if filename:
        res.to_csv('reformated_{}'.format(filename))
    return res

def compute_measure(fname = 'data/processed/datasets.h5'):
    test_in, test_out, predict, lookup = load_testing_sets(fname)
    time_forward = 6

    intervals = extract_continuous_intervals(lookup)

    bincounts = np.zeros((time_forward,7))
    counter = 0
    for start, stop in intervals:
        counter += 1
        for i in range(time_forward):
            _, path, _ = dtw_measure(predict[start:stop, 0, i], test_out[start:stop, 0, i], time_forward)
            bins, counts = np.unique(abs(path[0, :] - path[1, :]), return_counts=True)
            bincounts[i, bins] += counts
            
    lat_res = pd.DataFrame(data=bincounts, index=np.arange(1, time_forward+1), columns=np.arange(7))
    ref_res = reformat_dtw_res(lat_res)
    print(ref_res)

if __name__ == '__main__':
    compute_measure()