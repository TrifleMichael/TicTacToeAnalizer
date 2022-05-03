import random

import numpy as np
import matplotlib.pyplot as plt
from random import uniform

# Hopfield networks can hold about 0.138 \* n_neurons for better denoising <br>
# 0.138 \* n_neurons = 0.138 \* 25 = 3.45 ~ 3 <br>
n_train = 3
n_test = 100

# no of images to show in output plot
n_train_disp = 8

# Size of image(width)
n_side = 5

# No of neurons
n_neurons = n_side * n_side

perfect_data = [
    [1, -1, -1, -1, 1,
     -1, 1, -1, 1, -1,
     -1, -1, 1, -1, -1,
     -1, 1, -1, 1, -1,
     1, -1, -1, -1, 1],

    [1, 1, 1, 1, 1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     1, 1, 1, 1, 1],

    [-1 for i in range(25)]
]


def distort(arr, distortionPercent, maxDistortions=9):
    output = arr[:]
    distortions = 0
    for i in range(len(output)):
        if random.uniform(0, 1) <= distortionPercent:
            output[i] *= -1
            distortions += 1
        if distortions >= maxDistortions:
            break
    return output


# Function to train the network using Hebbian learning rule
def train(neu, training_data):
    w = np.zeros([neu, neu])
    for data in training_data:
        w += np.outer(data, data)
    for diag in range(neu):
        w[diag][diag] = 0
    return w


# Function to test the network
def test(weights, testing_data):
    success = 0.0

    output_data = []

    for data in testing_data:
        true_data = data[0]
        noisy_data = data[1]
        predicted_data = retrieve_pattern(weights, noisy_data)
        if np.array_equal(true_data, predicted_data):
            success += 1.0
        output_data.append([true_data, noisy_data, predicted_data])

    return (success / len(testing_data)), output_data


# Function to retrieve individual noisy patterns
def retrieve_pattern(weights, data, steps=10):
    res = np.array(data)

    for _ in range(steps):
        for i in range(len(res)):
            raw_v = np.dot(weights[i], res)
            if raw_v > 0:
                res[i] = 1
            else:
                res[i] = -1
    return res


# function to plot the images after during testing phase
def plot_images(images, title, no_i_x, no_i_y=3):
    fig = plt.figure(figsize=(10, 15))
    fig.canvas.set_window_title(title)
    images = np.array(images).reshape(-1, 5, 5)
    images = np.pad(
        images, ((0, 0), (1, 1), (1, 1)), 'constant', constant_values=-1)
    for i in range(no_i_x):
        for j in range(no_i_y):
            if no_i_x * j + i >= len(images):
                break
            ax = fig.add_subplot(no_i_x, no_i_y, no_i_x * j + (i + 1))
            ax.matshow(images[no_i_x * j + i], cmap="gray")
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))

            if j == 0 and i == 0:
                ax.set_title("Real")
            elif j == 0 and i == 1:
                ax.set_title("Distorted")
            elif j == 0 and i == 2:
                ax.set_title("Reconstructed")


# Train
train_data = [np.array(d) for d in perfect_data][:n_train]
W = train(n_neurons, train_data)
# Test
test_data = [[data, distort(data, 0.10, 25)] for data in perfect_data]
accuracy, op_imgs = test(W, test_data)

# Print accuracy
print("Accuracy of the network is %f" % (accuracy * 100))

# Plot test result
plot_images(op_imgs, "Reconstructed Data", n_train_disp)
plt.show()
