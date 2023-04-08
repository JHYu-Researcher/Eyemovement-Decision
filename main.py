import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import nn
from Model.model import DoubleLayerModel
from sklearn import preprocessing
from scipy import stats

def preprocess_data(csv_file):
    data = pd.read_csv(csv_file)
    data = data.drop(data.columns[[9,10,13,15]],axis=1)
    data["label"] = pd.factorize(data.Stimulus)[0]
    X = data.iloc[:,6:12]
    Y = data.label
    index = (np.abs(stats.zscore(X)) < 3).all(axis=1)
    X = X[index]
    Y = Y[index]
    return X,Y

def train(epoch, trainloader, testloader):
    correct = 0
    total = 0
    running_loss = 0
    model.train()
    for x, y in trainloader:
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)
            correct += (y_pred == y).sum().item()
            total += y.size(0)
            running_loss += loss.item()

    epoch_loss = running_loss / len(trainloader.dataset)
    epoch_acc = correct / total
    test_correct = 0
    test_total = 0
    test_running_loss = 0
    model.eval()
    with torch.no_grad():
        for x, y in testloader:
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            y_pred = torch.argmax(y_pred, dim=1)
            test_correct += (y_pred == y).sum().item()
            test_total += y.size(0)
            test_running_loss += loss.item()
    epoch_test_loss = test_running_loss / len(testloader.dataset)
    epoch_test_acc = test_correct / test_total

    if epoch % log_freq == 0:
        print('epoch:', epoch,
              '  loss:', round(epoch_loss, 3),
              '  accuracy:', round(epoch_acc, 3),
              '  test_loss:', round(epoch_test_loss, 3),
              '  test_accuracy:', round(epoch_test_acc, 3)
              )
    return epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc

def make_dataloader(path, test_dir):
    train_X, train_Y = pd.DataFrame(), pd.DataFrame()
    test_X, test_Y = pd.DataFrame(), pd.DataFrame()
    for p in os.listdir(path):
        csv_file = os.path.join(path, p, "data.csv")
        if p != test_dir:
            tmp_X, tmp_Y = preprocess_data(csv_file)
            train_X = pd.concat([train_X, tmp_X], ignore_index=True)
            train_Y = pd.concat([train_Y, tmp_Y], ignore_index=True)
        else:
            tmp_X, tmp_Y = preprocess_data(csv_file)
            test_X = pd.concat([test_X, tmp_X], ignore_index=True)
            test_Y = pd.concat([test_Y, tmp_Y], ignore_index=True)

    # standardization
    scaler = preprocessing.StandardScaler().fit(train_X)
    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)
    train_Y = train_Y.values.reshape(-1)
    test_Y = test_Y.values.reshape(-1)

    # dataloader
    train_X = torch.from_numpy(train_X).type(torch.float32)
    train_Y = torch.from_numpy(train_Y).type(torch.int64)

    test_X = torch.from_numpy(test_X).type(torch.float32)
    test_Y = torch.from_numpy(test_Y).type(torch.LongTensor)

    train_ds = TensorDataset(train_X, train_Y)
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True)

    test_ds = TensorDataset(test_X, test_Y)
    test_dl = DataLoader(test_ds, batch_size=batch)

    return train_dl, test_dl, scaler


if __name__ == '__main__':
    # hyper param
    batch = 10
    epochs = 800
    learning_rate = 1e-2
    log_freq = 20
    data_path = "data/"
    test_acc_all = []
    for p in os.listdir(data_path):
        model = DoubleLayerModel()
        opt = torch.optim.SGD(model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        train_dl, test_dl, _ = make_dataloader(data_path, p)
        test_epoch_acc = []
        for epoch in range(epochs):
            _, _, _, epoch_test_acc = train(epoch, train_dl, test_dl)
            test_epoch_acc.append(epoch_test_acc)
        test_acc_all.append(max(test_epoch_acc))
    print(np.mean(test_acc_all))
