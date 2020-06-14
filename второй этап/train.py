import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import random
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
import lib

history = pd.read_csv('data/history.tsv', sep='\t')
users = pd.read_csv('data/users.tsv', sep='\t')
validate = pd.read_csv('data/validate.tsv', sep='\t')
validate_answers = pd.read_csv('data/validate_answers.tsv', sep='\t')

df = validate.copy()
df['hours'] = df.hour_end - df.hour_start
df['at_least_one'] = validate_answers.at_least_one.copy()
df['at_least_two'] = validate_answers.at_least_two.copy()
df['at_least_three'] = validate_answers.at_least_three.copy()

user_saw_ads = history.groupby('user_id').size()
user_publishers = np.zeros((27768 + 1, 22))
for i, row in history.iterrows():
    user_publishers[int(row.user_id)][int(row.publisher)] += 1.0 / user_saw_ads[row.user_id]

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
np.save("./final_solve/user_publishers", user_publishers)
df['publ_fit_users'] = df.apply(make_publ_fit_users, axis=1)

users_overall_active_alt = history.groupby('user_id').size().reindex(range(27768 + 1), fill_value=0) / 1488
def make_users_overall_activity(users_in_row):
    tmp = pd.Series(users_in_row.split(',')).apply(int)
    feater = []
    for current_user in tmp:
        feater.append(users_overall_active_alt[current_user])
    return np.array(feater)
users_overall_active_alt.to_pickle("./final_solve/users_overall_active")
df['users_overall_activity'] = df.user_ids.apply(make_users_overall_activity)

df.cpm = (df.cpm - df.cpm.mean()) / df.cpm.std()
df.hours = (df.hours - df.hours.mean()) / df.hours.std()

final_df = df[['cpm', 'hours', 'users_overall_activity', 'publ_fit_users',
               'at_least_one','at_least_two', 'at_least_three']].to_numpy()
num_featers = 4
X, y = final_df[:, 0:num_featers], final_df[:, num_featers:]
y = torch.Tensor(y.astype('float32')).float()
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#y_train = torch.Tensor(y_train.astype('float32')).float()
#y_test = torch.Tensor(y_test.astype('float32')).float()

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

X, ids = expand_df(X)
X[:, 2:] = scale(X[:, 2:])
X = torch.Tensor(X)
"""
X_train, train_ids = expand_df(X_train)
X_test, test_ids = expand_df(X_test)
X_train[:, 2:] = scale(X_train[:, 2:])
X_test[:, 2:] = scale(X_test[:, 2:])
X_train = torch.Tensor(X_train)
X_test = torch.Tensor(X_test)
"""

model = lib.Net(4, [12, 12, 12, 12])
opt = optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.0001)
batch_size = 32
epochs = 35
test_loss = []
train_loss = []
overfit_loss = []
for i in range(epochs):
    #order = np.random.permutation(y_train.shape[0])
    order = np.random.permutation(y.shape[0])
    batch_loss = torch.Tensor(batch_size, 3)
    target_scores = torch.Tensor(batch_size, 3)
    for it, j in enumerate(order):
        #X_current, y_current = lib.getitem(X_train, y_train, train_ids, j)
        X_current, y_current = lib.getitem(X, y, ids, j)
        scores = model(X_current)
        batch_loss[it % batch_size] = scores.mean(axis=0)
        target_scores[it % batch_size] = y_current
        if (it + 1) % batch_size == 0:
            opt.zero_grad()
            loss = lib.get_smoothed_mean_log_accuracy_ratio(target_scores, batch_loss)  # for custom
            batch_loss = torch.Tensor(batch_size, 3)
            target_scores = torch.Tensor(batch_size, 3)
            loss.backward()
            opt.step()

    #epoch_loss = torch.Tensor(y_train.shape[0], 3)
    epoch_loss = torch.Tensor(y.shape[0], 3)
    for j in range(y.shape[0]):
    #for j in range(y_train.shape[0]):
        #X_current, y_current = lib.getitem(X_train, y_train, train_ids, j)
        X_current, y_current = lib.getitem(X, y, ids, j)
        scores = model(X_current)
        epoch_loss[j] = scores.mean(axis=0)

    #train_loss.append(int(lib.get_smoothed_mean_log_accuracy_ratio(y_train, epoch_loss).item()))
    overfit_loss.append(int(lib.get_smoothed_mean_log_accuracy_ratio(y, epoch_loss).item()))
    """
    epoch_loss = torch.Tensor(y_test.shape[0], 3)
    for j in range(y_test.shape[0]):
        X_current, y_current = lib.getitem(X_test, y_test, test_ids, j)
        scores = model(X_current)
        epoch_loss[j] = scores.mean(axis=0)
    test_loss.append(int(lib.get_smoothed_mean_log_accuracy_ratio(y_test, epoch_loss).item()))
    """
#print(train_loss)
#print(test_loss)
print(overfit_loss)
torch.save(model.state_dict(), "final_solve/weights.pt")