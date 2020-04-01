import sys
import pandas as pd
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sys.path.append('.')

from src.data.preprocess import preprocess_raw

def extract_from_split(split, train_ind, test_ind):
    merged = None
    tr_ind = split.copy()
    te_ind = split.copy()
    train_ind.sort()
    test_ind.sort()
    for ind in test_ind[::-1]:
        del tr_ind[ind]
    for ind in train_ind[::-1]:
        del te_ind[ind]
    return pd.concat(tr_ind).sort_index(), pd.concat(te_ind).sort_index()

def controlled_train_test_split(data):
    '''Input:
        data: panda dataframe with dates
       Output:
        test set: The month July and December of each year
        train set: The remaining months
    '''
    split = [g for n,g in data.groupby(pd.Grouper(freq='M')) if g.shape[0] != 0]

    test_ind = np.hstack([[3+i*12, 7+i*12, 11+i*12] for i in range(int(len(split)/12))])
    train_ind = list(filter(lambda x: x not in test_ind, np.arange(len(split))))

    train, test = extract_from_split(split, train_ind, test_ind)

    return train, test

def split_data(data, test_size, freq='M'):
    '''Input:
        input_, output_: numpy matrices that need to be randomly split on the first index
        test_size, train_size, valid_size: percentages that sum to 1
    '''
    split = [g for n,g in data.groupby(pd.Grouper(freq=freq)) if g.shape[0] != 0]
    random = 10321
    train_ind, test_ind = train_test_split(np.arange(len(split)), random_state=random, test_size=test_size)
    train, test = extract_from_split(split, train_ind, test_ind)

    return train, test

def extract_data(data, features, output):
    if type(features) is not list:
        features = [features]
    if type(output) is not list:
        output = [output]

    data_in = data[features].copy()
    data_out = data[output].shift(-1).copy()
    return data_in, data_out

def shift_and_normalize(data, scaler=None):
    errors = data[data.isna().any(axis=1)]
    clean = data.dropna(axis=0)
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(clean.values)
    scaled_values = scaler.transform(clean.values)
    clean = pd.DataFrame(data=scaled_values, index=clean.index, columns=clean.columns)
    return pd.concat([clean, errors]).sort_index(), scaler

def preprocess_data(data, scaler=None):
    if scaler is None:
        data, scaler = shift_and_normalize(data)
    else:
        data, _ = shift_and_normalize(data, scaler)

    return data, scaler

def format_to_lstm_input(input_, output_, time_back, time_forward):
    num_examples = input_.shape[0]
    size = num_examples - time_forward - time_back + 1
    num_features = input_.shape[1]
    lookup = np.zeros((size, time_forward), dtype='datetime64[s]')
    X = np.zeros((size, time_back, num_features))
    y = np.zeros((size, output_.shape[1], time_forward))
    valid_ins = input_.iloc[:,0].rolling(str(time_back)+'h').apply(lambda x: True if x.shape[0] == time_back else False, raw=True)
    dates = input_.index.values
    input_ = input_.values
    output_ = output_.values
    ind = 0
    for i, val in valid_ins.reset_index(drop=True).iteritems():
        if val != 1:
            continue
        j = i + 1
        X[ind] = input_[j-time_back:j, :]
        out = output_[i:i+time_forward, :].T                
        if out.shape[1] != time_forward:
            continue
        y[ind] = out
        lookup[ind] = dates[i:i+time_forward] # Store outputdates
        if not np.isnan(X[ind]).any():
            if not np.isnan(y[ind]).any():
                ind += 1

    return X[:ind], y[:ind], lookup[:ind]

def get_storm_dates(stormname='data/external/semi_auto_storms_8114.csv'):
    return pd.read_csv(stormname, index_col=0, dtype={'date1': str, 'date2': str},
                       parse_dates=['date1', 'date2'])

def get_storms(data, storm_dates):
    '''Check which storms lie in the given dataset
    Input:
        data: dataset with dates
        storm_dates: list of known storm occurences
    Output:
        measured features during the storm
        storm-dates that lie in data
    '''
    rs = []
    valid_storms = []
    for (_, dates) in storm_dates.iterrows():
        ss = data.loc[dates[0]:dates[1]]
        if ss.shape[0] != 0:
            rs.append(ss)
            valid_storms.append((dates[0].strftime('%Y-%m-%d'), dates[1].strftime('%Y-%m-%d')))
    print('Number of storms in dataset: {}'.format(len(valid_storms)))
    return pd.concat(rs), np.array(valid_storms)

def store_data_sets(train_in, train_out, valid_in, valid_out, test_in, test_out, lookup, test_storm_dates, fname='data/processed/datasets.h5'):
    with h5py.File(fname, 'w') as f:
        train = f.create_group('train_sets')
        train.create_dataset('train_in', data=train_in)
        train.create_dataset('train_out', data=train_out)
        valid = f.create_group('valid_sets')
        valid.create_dataset('valid_in', data=valid_in)
        valid.create_dataset('valid_out', data=valid_out)
        test = f.create_group('test_sets')
        test.create_dataset('test_in', data=test_in)
        test.create_dataset('test_out', data=test_out)
        test.create_dataset('lookup', data=lookup.astype(np.long), dtype='int64')
        storms = test.create_group('storms')
        storms.create_dataset('storm_dates', data=test_storm_dates.astype('S'))
    

def build_input(features, output, time_back, time_forward, procf='data/interim/data.h5', input_f='data/processed/dataset.h5'):
    startdate = '14-01-2001'
    enddate = '01-01-2016'

    try:
        data = pd.read_hdf(procf, 'data')
    except FileNotFoundError as err:
        print(err)
        print('Building new preprocessed file')
        preprocess_raw(path_proc=procf)
        data = pd.read_hdf(procf, 'data')

    data = data[startdate:enddate]

    train, test = controlled_train_test_split(data)
    train, valid = split_data(train, 0.2, 'M')
    train_in, train_out = extract_data(train, features, output)
    valid_in, valid_out = extract_data(valid, features, output)
    test_in, test_out = extract_data(test, features, output)

    tr_in, sclr = preprocess_data(train_in)
    val_in, _ = preprocess_data(valid_in, sclr)
    t_in, _ = preprocess_data(test_in, sclr)

    train_in, train_out, _ = format_to_lstm_input(tr_in, train_out, time_back, time_forward)
    valid_in, valid_out, _ = format_to_lstm_input(val_in, valid_out, time_back, time_forward)
    test_in, test_out, lookup = format_to_lstm_input(t_in, test_out, time_back, time_forward)
        
    storm_dates = get_storm_dates()
    test_storms, test_storm_dates = get_storms(test, storm_dates)

    store_data_sets(train_in, train_out, valid_in, valid_out, test_in, test_out, lookup, test_storm_dates, fname=input_f)


if __name__ == '__main__':
    time_forward = 6
    time_back = 6
    features = ['Dst', '|B|', 'Bz_GSM', 'SWDens', 'SWSpeed']
    output = 'Dst'
    build_input(features, output, time_back, time_forward)


