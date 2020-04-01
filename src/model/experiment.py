import sys
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

sys.path.append('.')

from src.data.build_input import build_input
from src.model.LSTMnn import LSTMnn
from src.model.metrics import evaluate
from src.visualize.visualize import plot_set_of_storms

# data loading and storing

def load_training_sets(fname='data/processed/datasets2.h5'):
    with h5py.File(fname, 'r') as f:
        train_in = f['train_sets/train_in'][:]
        train_out = f['train_sets/train_out'][:]
        valid_in = f['valid_sets/valid_in'][:]
        valid_out = f['valid_sets/valid_out'][:]

    return train_in, train_out, valid_in, valid_out

def load_testing_sets(fname='data/processed/datasets.h5'):
    with h5py.File(fname, 'r') as f:
        test_in = f['test_sets/test_in'][:]
        test_out = f['test_sets/test_out'][:]
        lookup = f['test_sets/lookup'][:]
    return test_in, test_out, lookup.astype('datetime64[s]')

def load_test_storm_dates(fname='data/processed/datasets.h5'):
    with h5py.File(fname, 'r') as f:
        test_storm_dates = f['test_sets/storms/storm_dates'][:]
    return test_storm_dates.astype('datetime64')

def numpy_to_dataloader(input_, output_, batch_size=64):
    set_ = TensorDataset(torch.Tensor(input_), torch.Tensor(output_))
    return DataLoader(set_, batch_size=batch_size, shuffle=True)

def save_test_forecast(predict, fname='data/processed/datasets.h5'):
    with h5py.File(fname, 'a') as f:
        try:
            f['test_sets'].create_dataset('prediction', data=predict)
        except RuntimeError:
            del f['test_sets/prediction']
            f['test_sets'].create_dataset('prediction', data=predict)
        

# model training

def train_model(model, epochs, training, validation, criterion, optimizer, file_path=None, verbal=False):
    '''Train the model, using validation set for best selection
    The model is trained on the training data. Each epoch, the model is than used to predict the validation data.
    The best iteration of the model is stored and returned after going through every epoch.
    If a file_path is given, the best model configuration is stored in a file.
    '''
    min_val = np.inf
    best_loss = np.inf
    best_epoch = 0
    best_model = None
    for t in range(epochs):
        running_loss = 0
        model.train()
        for x, y in training:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        if verbal:
            print('Epoch {}: Training loss: {}'.format(t, running_loss/len(training.dataset)))

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for val_in, val_out in validation:
                y_pred = model(val_in)
                loss = criterion(val_out, y_pred)
                val_loss += loss.item()
            if verbal:
                print('Validation loss: ', val_loss / len(validation.dataset))
            if val_loss < min_val:
                min_val = val_loss
                best_loss = running_loss/len(training.dataset)
                best_epoch = t
                best_model = model.state_dict()
    if file_path is not None:
        torch.save(best_model, file_path)
    print('Best loss: {}, Min validation loss: {}, Epoch: {}'.format(best_loss, min_val, best_epoch))

    model.load_state_dict(best_model)

def run_model(model, data_in, file_path=None):
    if file_path is not None:
        model.load_state_dict(torch.load(file_path))
    model.eval() 
    pred = model(torch.Tensor(data_in))
    return pred.detach().numpy()

def main():
    import os.path

    time_forward = 6
    time_back = 6
    features = ['Dst', '|B|', 'Bz_GSM', 'SWDens', 'SWSpeed']
    output = 'Dst'
    num_epochs = 30
    hidden_size = 50
    momentum = 0.8
    num_layers = 1
    learning_rate = 0.0003
    batch_size = 64
    num_feat = len(features)

    datafile = 'data/processed/datasets.h5'

    if not os.path.isfile(datafile):
        print('Datafile does not exist, creating one now...')
        build_input(features, output, time_back, time_forward, input_f=datafile)  
        print('Done')

    train_in, train_out, valid_in, valid_out = load_training_sets(datafile)
    train = numpy_to_dataloader(train_in, train_out, batch_size)
    valid = numpy_to_dataloader(valid_in, valid_out, 1024)

    # model init
    model = LSTMnn(num_feat, num_layers, hidden_size, time_forward)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, momentum=momentum)

    # model training
    # training takes a long time. Only train when storing to a new file.
    model_file = 'models/experiment'
    if not os.path.isfile(model_file):
        train_model(model, num_epochs, train, valid, criterion, optimizer, verbal=True, file_path=model_file)

    # model testing
    test_in, test_out, lookup = load_testing_sets(datafile)
    predict = run_model(model, test_in, file_path=model_file)

    # evaluation
    ev = evaluate(predict[:, 0], test_out[:, 0])
    print(pd.DataFrame(data=ev, index=['t+{}'.format(i+1) for i in range(6)]))

    ## storm visualization
    test_storm_dates = load_test_storm_dates(datafile)
    times = [0, 2, 4]
    selected_storms = [test_storm_dates[0], test_storm_dates[7], test_storm_dates[13]]
    plot_set_of_storms(test_out[:, 0], predict[:, 0], lookup, selected_storms, times, save=True, fname='figures/experiment')

    # store prediction to file
    save_test_forecast(predict, datafile)

if __name__ == '__main__':
    main()