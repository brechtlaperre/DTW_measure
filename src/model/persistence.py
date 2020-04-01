import sys
import numpy as np
import pandas as pd

sys.path.append('.')

from src.data.build_input import controlled_train_test_split
from src.dtw.dtw_measure import dtw_measure
from src.model.metrics import evaluate

def persistence_predict(data, time):
    '''Forecast a given feature for a given forecast time
    Input:
        data: pandas dataframe containing all the to be forecasted features
        time: time to be forecasted
    Output:
        res: pandas dataframe 
    '''
    res = data.shift(time)
    return res

def persistence_dtw_measure(data, time_forward):
    # Allow only one feature at the time
    assert(data.shape[1] == 1)
    
    pers = data.copy()
    for i in range(time_forward):
        pers['T_{}'.format(i+1)] = persistence_predict(data, i+1)
    pers = pers.dropna() # remove NaN-values
    intervals = extract_cont_intervals_from_index(pers.index)
    
    bincounts = np.zeros((time_forward,7))
    length = intervals.shape[0]
    for num, (start, stop) in enumerate(intervals):
        print('{} out of {}'.format(num+1, length))
        month = pers[start:stop]
        for i in range(time_forward):
            # dtw_measure(forecast, truth, warping path)
            _, path, _ = dtw_measure(month['T_{}'.format(i+1)].to_numpy(), month.iloc[:,0].to_numpy(), 6)
            bins, counts = np.unique(abs(path[0, :] - path[1, :]), return_counts=True)
            bincounts[i, bins] += counts

    bincounts = pd.DataFrame(data=bincounts, index=np.arange(1, time_forward+1), columns=np.arange(7))
    return bincounts

def persistence_eval(features, time_forward, dtw=True):
    r'''Evaluation of the persistence model. 
    This model does the standard metric test, together with a dtw count. 
    The dtw count keeps into consideration discontinuities, splitting the data
    in continuous pieces first.
    Evaluates times [1, 2, ..., time_forward]
    Input:
        data: Pandas dataframe with DateTime index and to be forecasted features
        time_forward: Number of hours evaluated
        dtw: boolean, run dtw measure when true
    Output:
        dtw-result is written to a file directly
        res: Metric evaluation
    '''
    data_all = np.repeat(features.to_numpy()[time_forward+1:-time_forward], time_forward, axis=1)
    pers_all = np.zeros(data_all.shape)
    for i, t in enumerate(range(1, 1+time_forward)):
        persist = persistence_predict(features, t)
        pers_all[:, t-1] = persist.to_numpy()[time_forward+1:-time_forward, 0]
        i += 1
    res = evaluate(pers_all, data_all)

    bincounts = None
    if dtw:
        bincounts = persistence_dtw_measure(features, time_forward)        

    return res, bincounts

def extract_cont_intervals_from_index(index):
    r'''Check lookup table for time discontinuities
    output: 
        Returns list of continouos times inside the lookup table
    '''
    min_size = 10
    timeseries = []
    p = True
    series = index
    
    while len(series) > 0:
        # We can assume that the series starts from non-missing values, so the first diff gives sizes of continous intervals
        diff = pd.date_range(series[0], series[-1], freq='H').difference(series)
        if len(diff) > 0:
            if pd.Timedelta(diff[0] - pd.Timedelta('1h') - series[0])/pd.Timedelta('1h') > min_size:
                v1 = np.datetime64(series[0])
                v2 = np.datetime64(diff[0] - pd.Timedelta('1h'))
                timeseries.append([v1, v2])
            if pd.Timedelta(series[-1] - diff[-1] - pd.Timedelta('1h'))/pd.Timedelta('1h') > min_size:
                v1 = np.datetime64(diff[-1] + pd.Timedelta('1h'))
                v2 = np.datetime64(series[-1])
                timeseries.append([v1, v2])
            diff = pd.date_range(diff[0], diff[-1], freq='H').difference(diff)
        else:
            # Only when diff is empty
            v1 = np.datetime64(series[0])
            v2 = np.datetime64(series[-1])
            timeseries.append([v1, v2])
        series = diff
        
    return np.array(timeseries)

def persistence_experiment():
    startdate = '14-01-2001'
    enddate = '01-01-2016'

    data = pd.read_hdf('data/interim/data.h5', 'data')
    data = data[startdate:enddate]
    _, test = controlled_train_test_split(data)
    output = 'Dst'
    time_forward = 6

    pers_res, bincounts = persistence_eval(test[[output]], time_forward)

    return pers_res, bincounts

if __name__ == '__main__':
    pers_res, bincounts = persistence_experiment()

    print(pers_res)