import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
import os.path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import random
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
import sys
import lib

# 0 - script name, 1 - model, 2... - featers
def main():
    validate = pd.read_csv('/tmp/data/test.tsv', sep="\t")
    #validate = pd.read_csv('validate.tsv', sep="\t") #COPY data/validate.tsv validate.tsv
    #validate_answers = pd.read_csv('data/validate_answers.tsv', sep='\t')
    #y = torch.Tensor(validate_answers[['at_least_one','at_least_two', 'at_least_three']].to_numpy().astype('float32')).float()
    #users = pd.read_csv(sys.argv[5], sep='\t') #'data/users.tsv'

    model = lib.Net(4, [12, 12, 12, 12])
    model.load_state_dict(torch.load(sys.argv[1]))
    model.eval()

    user_publishers = np.load(sys.argv[2])
    users_overall_active = pd.read_pickle(sys.argv[3])

    def make_users_overall_activity(users_in_row):
        tmp = pd.Series(users_in_row.split(',')).apply(int)
        feater = []
        for current_user in tmp:
            feater.append(users_overall_active[current_user])
        return np.array(feater)
    validate['users_overall_activity'] = validate.user_ids.apply(make_users_overall_activity)
    validate['hours'] = validate.hour_end - validate.hour_start

    def make_publ_fit_users(row):
        users_in_row = pd.Series(row.user_ids.split(',')).apply(int)
        publishers_in_row = pd.Series(row.publishers.split(',')).apply(int)

        feater = []
        for current_user in users_in_row:
            user_sum = 0
            for current_publisher in publishers_in_row:
                user_sum += user_publishers[current_user][current_publisher]
            feater.append(user_sum)
        return np.array(feater)

    validate['publ_fit_users'] = validate.apply(make_publ_fit_users, axis=1)

    def expand_df(cur_df):
        ids = []
        series = []
        current_id = 0
        for i in range(cur_df.shape[0]):
            len_ = cur_df[i][2].size
            for j in range(len_):
                series.append([cur_df[i][0], cur_df[i][1], cur_df[i][2][j], cur_df[i][3][j]])
            ids.append((current_id, current_id + len_))
            current_id += len_
        series = np.array(series)
        ids = pd.Series(ids)
        return series, ids

    validate.cpm = (validate.cpm - validate.cpm.mean()) / validate.cpm.std()
    validate.hours = (validate.hours - validate.hours.mean()) / validate.hours.std()

    X = validate[['cpm', 'hours', 'users_overall_activity', 'publ_fit_users']].to_numpy()
    X, ids = expand_df(X)
    X[:, 2:] = scale(X[:, 2:])
    X = torch.tensor(X).float()

    loss = torch.Tensor(validate.shape[0], 3)
    for j in range(validate.shape[0]):
        X_current, y_current = lib.getitem(X, None, ids, j)
        scores = model(X_current)
        loss[j] = scores.mean(axis=0)
    #print(int(lib.get_smoothed_mean_log_accuracy_ratio(y, loss).item()))
    tmp = loss.detach().numpy()
    tmp = pd.DataFrame({'at_least_one':tmp[:, 0], 'at_least_two':tmp[:, 1], 'at_least_three':tmp[:, 2]})
    tmp.to_csv('/opt/results/result.tsv', sep="\t", index=False, header=True)
    #tmp.to_csv('result.tsv', sep="\t", index=False, header=True)
    #print(tmp.shape)
    #print(os.path.isfile('result.tsv'))

if __name__ == '__main__':
    #start = time.time()
    main()
    #print(time.time() - start)