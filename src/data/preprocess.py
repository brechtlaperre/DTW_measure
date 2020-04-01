import sys
import pandas as pd
import numpy as np

sys.path.append('.')

def read_raw_data(filepath):
    return pd.read_csv(filepath, index_col=0)

def add_features(data):
     data['Delta_Dst'] = data.Dst.diff()
     return data

def compose_date(years, months=1, days=1, weeks=None, hours=None, minutes=None,
                 seconds=None, milliseconds=None, microseconds=None, nanoseconds=None):
    '''Create timestamp index from raw data
    '''
    years = np.asarray(years) - 1970
    months = np.asarray(months) - 1
    days = np.asarray(days) - 1
    types = ('<M8[Y]', '<m8[M]', '<m8[D]', '<m8[W]', '<m8[h]',
             '<m8[m]', '<m8[s]', '<m8[ms]', '<m8[us]', '<m8[ns]')
    vals = (years, months, days, weeks, hours, minutes, seconds,
            milliseconds, microseconds, nanoseconds)
    return sum(np.asarray(v, dtype=t) for t, v in zip(types, vals)
               if v is not None)


def preprocess(data):
    '''Preprocess data
    '''
    data.index = compose_date(data['YEAR'], days=data['DOY'], hours=data['Hour'])
    data = data.drop(columns=['YEAR', 'DOY', 'Hour'])
    data = data.drop(columns=['pc', 'Pflux1', 'Pflux2', 'Pflux4'])
    data = add_features(data)
    return data


def preprocess_raw(path_raw='data/raw/features.csv', path_proc='data/interim/processed.h5'):
    raw = read_raw_data(path_raw)
    processed = preprocess(raw)
    processed.to_hdf(path_proc, 'data')
    


if __name__ == '__main__':
    preprocess_raw()

