
import numpy as np
import csv
import pandas as pd
from argparse import Namespace

seed = 9487
np.random.seed(seed)

def valid(x, y):
    return y <= 20  

def parse2train(data, feats):
    x = []
    y = []
    total_length = data.shape[1] - 8

    for i in range(total_length):
        x_tmp = data[feats, i:i+8]
        y_tmp = data[-1, i+8] 

        if valid(x_tmp, y_tmp):
            x.append(x_tmp.reshape(-1,))
            y.append(y_tmp)

    x = np.array(x)
    y = np.array(y)
    return x, y

def parse2test(data, feats):
    x = []
    for i in range(90):
        x_tmp = data[feats, 8*i: 8*i+8]
        x.append(x_tmp.reshape(-1,))
    x = np.array(x)

    # Add polynomial (squared) features for testing, as done during training
    x_poly = np.hstack((x, x**2))
    return x_poly

def minibatch_2(x, y, config):
    x_poly = np.hstack((x, x**2))

    index = np.arange(x_poly.shape[0])
    np.random.shuffle(index)
    x_poly = x_poly[index]
    y = y[index]

    batch_size = config.batch_size
    lr = config.lr
    lam = config.lam
    epoch = config.epoch
    decay_rate = config.decay_rate
    epsilon = 1e-8

    w = np.full((x_poly.shape[1], 1), 0.1)
    bias = 0.1

    cache_w = np.zeros_like(w)
    cache_b = 0.0

    for num in range(epoch):
        for b in range(int(x_poly.shape[0] / batch_size)):
            x_batch = x_poly[b * batch_size:(b + 1) * batch_size]
            y_batch = y[b * batch_size:(b + 1) * batch_size].reshape(-1, 1)

            pred = np.dot(x_batch, w) + bias
            loss = y_batch - pred

            g_t = np.dot(x_batch.T, loss) * (-2) / batch_size + 2 * lam * w
            g_t_b = loss.sum(axis=0) * (-2) / batch_size

            cache_w = decay_rate * cache_w + (1 - decay_rate) * g_t**2
            cache_b = decay_rate * cache_b + (1 - decay_rate) * g_t_b**2

            w -= lr * g_t / (np.sqrt(cache_w) + epsilon)
            bias -= lr * g_t_b / (np.sqrt(cache_b) + epsilon)

    return w, bias

train_config = Namespace(
    batch_size = 32,
    lr = 0.001,
    lam = 0.0001,
    epoch = 10000,
    decay_rate = 0.05
)

feats = [4,7,10]  

train_data_path = '/home/md703/Julie/ML/train.csv'
data = pd.read_csv(train_data_path)

data = data.values
train_data = np.transpose(np.array(np.float64(data)))
train_x, train_y = parse2train(train_data, feats)

w, bias = minibatch_2(train_x, train_y, train_config)
print("Trained model parameters (weights):", w)
print("Trained model bias:", bias)


test_data_path = '/home/md703/Julie/ML/test.csv'
data = pd.read_csv(test_data_path)
data = data.values
test_data = np.transpose(np.array(np.float64(data)))
test_x = parse2test(test_data, feats)

with open('my_sol.csv', 'w', newline='') as csvf:
    writer = csv.writer(csvf)
    writer.writerow(['Id','Predicted'])

    print(test_x.shape)
    for i in range(int(test_x.shape[0])):
        # Prediction of linear regression
        prediction = (np.dot(np.reshape(w, -1), test_x[i]) + bias)[0]
        writer.writerow([i, prediction])