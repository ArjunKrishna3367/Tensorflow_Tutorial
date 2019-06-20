from __future__ import absolute_import, division, print_function, unicode_literals

import pickle
import copy
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

import numpy as np
np.random.seed(7)


def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def get_data(index):
    if index == 'test':
        data = unpickle("cifar-10-batches-py/test_batch")
    else:
        data = unpickle(f"cifar-10-batches-py/data_batch_{index}")
    return np.array(data[b'data']), np.array(data[b'labels'])


classNames = []
labelDict = unpickle("cifar-10-batches-py/batches.meta")
for label in labelDict[b'label_names']:
    classNames.append(label.decode("utf-8"))

allImages = np.r_[get_data(1)[0], get_data(2)[0], get_data(3)[0], get_data(4)[0], get_data(5)[0]] / 255.0
allLabels = np.r_[get_data(1)[1], get_data(2)[1], get_data(3)[1], get_data(4)[1], get_data(5)[1]]
testImages, testLabels = get_data('test')[0] / 255.0, get_data('test')[1]


def conv_image(imgarr):
    temp = []
    temp2 = []
    temp3 = []
    for i in range(0, 1024):
        temp.append([imgarr[i], imgarr[i + 1024], imgarr[i + 2048]])
    for i in range(0, 32):
        for j in range(i * 32, (i + 1) * 32):
            temp2.append(temp[j])
        temp3.append(copy.copy(temp2))
        temp2.clear()
    return temp3

model = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adamax',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(allImages, allLabels, epochs=5, shuffle=True)

'''test model'''
print("\nTest:")
test_loss, test_acc = model.evaluate(testImages, testLabels)
print('Test accuracy:', test_acc)


predictions = model.predict(testImages)


# Method to plot image with predicted label
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], conv_image(img[i])
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(classNames[predicted_label],
                                         100 * np.max(predictions_array),
                                         classNames[true_label]),
               color=color)


# Method to plot bar chart of prediction probablities
def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks(range(10), classNames, rotation=90)
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


'''Plot image and chart of first image predictions'''
for i in range(25):
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plot_image(i, predictions, testLabels, testImages)
    plt.subplot(1,2,2)
    plot_value_array(i, predictions,  testLabels)
plt.show()




# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(conv_image(allImages[i]), cmap=plt.cm.binary)
#     plt.xlabel(classNames[allLabels[i]])
# plt.show()