import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_utils import batchify_data, run_epoch, train_model, Flatten
import utils_multiMNIST as U
path_to_data_dir = '../Datasets/'
use_mini_dataset = True

batch_size = 64
nb_classes = 10
nb_epoch = 30
num_classes = 10
img_rows, img_cols = 42, 28  # input image dimensions


class CNN(nn.Module):

    def __init__(self, input_dimension):
        super(CNN, self).__init__()
        self.flatten = Flatten()
        self.dropout = nn.Dropout()
        self.conv1 = nn.Conv2d(1, 32, (3, 3))
        self.conv2 = nn.Conv2d(32, 64, (3, 3))
        self.pool = nn.MaxPool2d((2, 2))
        self.lin1 = nn.Linear(2880, 512)
        self.lin2 = nn.Linear(512, 20)

    def forward(self, x):

        xf = F.leaky_relu(self.conv1(x))
        xf = self.pool(xf)
        xf = F.leaky_relu(self.conv2(xf))
        xf = self.pool(xf)
        xf = self.flatten(xf)
        xf = self.dropout(self.lin1(xf))

        xout = self.lin2(xf)
        out_first_digit = xout[:, :10]
        out_second_digit = xout[:, 10:]

        return out_first_digit, out_second_digit


def main():
    X_train, y_train, X_test, y_test = U.get_data(path_to_data_dir,
                                                  use_mini_dataset)

    # Split into train and dev
    dev_split_index = int(9 * len(X_train) / 10)
    X_dev = X_train[dev_split_index:]
    y_dev = [y_train[0][dev_split_index:], y_train[1][dev_split_index:]]
    X_train = X_train[:dev_split_index]
    y_train = [y_train[0][:dev_split_index], y_train[1][:dev_split_index]]

    permutation = np.array([i for i in range(len(X_train))])
    np.random.shuffle(permutation)
    X_train = [X_train[i] for i in permutation]
    y_train = [[y_train[0][i] for i in permutation],
               [y_train[1][i] for i in permutation]]

    # Split dataset into batches
    train_batches = batchify_data(X_train, y_train, batch_size)
    dev_batches = batchify_data(X_dev, y_dev, batch_size)
    test_batches = batchify_data(X_test, y_test, batch_size)

    # Load model
    input_dimension = img_rows * img_cols
    model = CNN(input_dimension)

    # Train
    train_model(train_batches, dev_batches, model)

    # Evaluate the model on test data
    loss, acc = run_epoch(test_batches, model.eval(), None)
    print('Test loss1: {:.6f}  accuracy1: {:.6f} \
          loss2: {:.6f}   accuracy2: {:.6f}'
          .format(loss[0], acc[0], loss[1], acc[1]))


if __name__ == '__main__':

    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)  # for reproducibility
    main()
