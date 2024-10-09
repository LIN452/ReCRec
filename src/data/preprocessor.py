"""
Codes for preprocessing real-world datasets used in the experiments
in the paper "Unbiased Recommender Learning from Missing-Not-At-Random Implicit Feedback".
"""
import codecs
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.model_selection import train_test_split


def preprocess_dataset(dataset_name:str = 'yahooR3',threshold: int = 4) -> Tuple:
    # load dataset.
    col = {0: 'user', 1: 'item', 2: 'rate'}
    with codecs.open(f'../data/{dataset_name}/train.txt', 'r', 'utf-8', errors='ignore') as f:
        data_train = pd.read_csv(f, delimiter=',', header=None)
        data_train.rename(columns=col, inplace=True)
    with codecs.open(f'../data/{dataset_name}/test.txt', 'r', 'utf-8', errors='ignore') as f:
        data_test = pd.read_csv(f, delimiter=',', header=None)
        data_test.rename(columns=col, inplace=True)
    num_users, num_items = data_train.user.max()+1, data_train.item.max()+1
    for _data in [data_train, data_test]:
        _data.user, _data.item = _data.user, _data.item 
        # binalize rating.
        _data.rate[_data.rate < threshold] = 0
        _data.rate[_data.rate >= threshold] = 1
    # train-val-test, split
    train, test = data_train.values, data_test.values
    valtemp,val = train_test_split(test,test_size=0.2,random_state=1234)
    _, item_freq = np.unique(train[:,1], return_counts=True)
    
    for i in range(num_items):
        item_freq[i] = np.sum(train[train[:, 2] == 1, 1]== i)
    
    pscore = (item_freq / item_freq.max()) ** 0.5
    exposure = train[:,:2]
    train = train[train[:, 2] == 1, :2]
    #val = val[val[:, 2] == 1, :2]
    
    # creating training data: including both missing data and negative data
    all_data = pd.DataFrame(
        np.zeros((num_users, num_items))).stack().reset_index()
    all_data = all_data.values[:, :2]
    
    unlabeled_data = np.array(
        list(set(map(tuple, all_data)) - set(map(tuple, train))), dtype=int)
    train = np.r_[np.c_[train, np.ones(train.shape[0])],
                  np.c_[unlabeled_data, np.zeros(unlabeled_data.shape[0])]]
    
    unlabeled_exposure_data = np.array(
        list(set(map(tuple, all_data)) - set(map(tuple, exposure))), dtype=int)
    exposure = np.r_[np.c_[exposure, np.ones(exposure.shape[0])],
                  np.c_[unlabeled_exposure_data, np.zeros(unlabeled_exposure_data.shape[0])]]

    # save datasets
    path = Path(f'../data/{dataset_name}/point')
    path.mkdir(parents=True, exist_ok=True)
    np.save(str(path / 'train.npy'), arr=train.astype(np.int))
    np.save(str(path / 'exposure.npy'), arr=exposure.astype(np.int))
    np.save(str(path / 'val.npy'), arr=val.astype(np.int))
    np.save(str(path / 'test.npy'), arr=test.astype(np.int))
    np.save(str(path / 'pscore.npy'), arr=pscore)
    np.save(str(path / 'item_freq.npy'), arr=item_freq)
