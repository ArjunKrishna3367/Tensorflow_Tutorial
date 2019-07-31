# import numpy as np
#
# numbers = []
# for i in range(3072):
#     numbers.append(i)
#
# arr = np.reshape(numbers, (-1, 3), order='F')

# arr = np.reshape(arr, (32, 32, 3))
#
# print (arr)

from __future__ import print_function
from keras.datasets import cifar10

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
