import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import picGenerator
# Hopfield networks can hold about 0.138 \* n_neurons for better denoising <br>
# 0.138 \* n_neurons = 0.138 \* 25 = 3.45 ~ 3 <br>
square_size = 15
# n_train = 7  # maybe more training better - Michal?
n_test = 100

# no of images to show in output plot
n_train_disp = 8
retrieve_steps = 10
# template = [
#     ['o','_','x'],
#     ['_','o','x'],
#     ['_','x','o']
# ]
# b = picGenerator.construct_board(square_size, broken=True, temp=template, alpha=0.3) #alpha <- [-1,1]
# picGenerator.save_image("./board_test.png", np.array(list(map(lambda x: ((1+x)//2*255, (1+x)//2*255, (1+x)//2*255), b)),
#                                                      dtype="i,i,i").astype(object).reshape((3*square_size+2,
#                                                                                             3*square_size+2)))

pixel_num = square_size * square_size
board_size = 3 * square_size + 2
n_train = int(board_size*board_size*0.138)

original_images = [
    picGenerator.generate_x(square_size),
    picGenerator.generate_o(square_size),
    picGenerator.generate_blank(square_size)
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
def train(training_data, neuron_num):
    weights = np.zeros([neuron_num, neuron_num])
    for data in training_data:
        weights += np.outer(data, data)
    for diag in range(neuron_num):
        weights[diag][diag] = 0
    return weights


# Function to test the network
def test(weights, testing_data):
    success = 0.0
    output_data = []
    for data in testing_data:
        true_data = data[0]
        noisy_data = data[1]
        predicted_data = predict_result(weights, noisy_data)
        if np.array_equal(true_data, predicted_data):
            success += 1.0
        output_data.append([true_data, noisy_data, predicted_data])

    return (success / len(testing_data)), output_data


# Function to retrieve individual noisy patterns
def predict_result(weights, data):
    result = np.array(data)
    for i in range(retrieve_steps):
        for j in range(len(result)):
            pixelVal = np.dot(weights[j], result)
            if pixelVal > 0:
                result[j] = 1
            else:
                result[j] = -1
    return result


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

#loads board from file
def load_board(path):
    board = np.array(Image.open(path).convert("L"))
    board = (((board // 255) - 0.5) * 2)
    return board

#predicts result for each square on board
def predict_board_result(board, original_images, weights):
    new_board = board.copy()

    board_results = [[-1 for _ in range(3)] for _ in range(3)]

    for i in range(0, board_size, square_size + 1):
        for j in range(0, board_size, square_size + 1):
            linearized_data = []
            for row in board[i:i + square_size, j:j + square_size]:
                for k in range(len(row)):
                    linearized_data.append(row[k])
            linearized_prediction = predict_result(weights, linearized_data)
            true_prediction = [[] for _ in range(square_size)]
            for k in range(0, square_size * square_size, square_size):
                true_prediction[k // square_size] = linearized_prediction[k:k + square_size]
            true_prediction = np.array(true_prediction)

            for symbol in range(len(original_images)):
                if np.array_equal(linearized_prediction, original_images[symbol]):
                    board_results[i // square_size][j // square_size] = symbol

            new_board[i:i + square_size, j:j + square_size] = true_prediction

    return new_board, board_results

def get_winner(board):
    winner = [False, False]
    for symbol in range(0, 2):
        for row in board:
            if row[0] == row[1] == row[2] == symbol:
                winner[symbol] = True

        for col in range(3):
            if board[0][col] == board[1][col] == board[2][col] == symbol:
                winner[symbol] = True

        if board[0][0] == board[1][1] == board[2][2] == symbol:
            winner[symbol] = True

        if board[2][0] == board[1][1] == board[0][2] == symbol:
            winner[symbol] = True

    if not winner[0] and not winner[1]:
        return "None"
    if winner[0] and not winner[1]:
        return "Cross"
    if not winner[0] and winner[1]:
        return "Circle"
    if winner[0] and winner[1]:
        return "Both"


def check_with_temp(template, board):
    d = {'x': 0, 'o': 1, '_': 2}
    return board == list(map(lambda row: list(map(lambda x: d[x], row)), template))


def generate_temp():
    return [[('x', 'o', '_')[random.randint(0, 2)] for _ in range(3)] for _ in range(3)]


def test_predicted_result():
    # template = [
    #     ['o', '_', 'x'],
    #     ['_', 'o', 'x'],
    #     ['_', 'x', 'o']
    # ]
    template = generate_temp()
    b = picGenerator.construct_board(square_size, broken=True, temp=template, alpha=0.3)  # alpha <- [-1,1]
    picGenerator.save_image("./board_test.png", np.array(
        list(map(lambda x: ((1 + x) // 2 * 255, (1 + x) // 2 * 255, (1 + x) // 2 * 255), b)),
        dtype="i,i,i").astype(object).reshape((3 * square_size + 2,
                                               3 * square_size + 2)))
    board = load_board("./board_test.png")
    new_board, board_results = predict_board_result(board, original_images, weights)
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(board)
    axs[1].imshow(new_board)
    axs[1].set_title("Winner: " + get_winner(board_results))
    plt.show()
    plt.close()
    return check_with_temp(template, board_results)



# Train
training_data = [np.array(d) for d in original_images][:n_train]
weights = train(training_data, pixel_num)
test_num = 30
print(f"accuracy: {sum(test_predicted_result() for _ in range(test_num))/test_num*100}%")

# board = load_board("./board_test.png")
# new_board, board_results = predict_board_result(board, original_images, weights)
#
# fig, axs = plt.subplots(1, 2)
# axs[0].imshow(board)
# axs[1].imshow(new_board)
# axs[1].set_title("Winner: " + get_winner(board_results))
# plt.show()
# print(check_with_temp(template, board_results))

"""
# Test
test_data = [[data, distort(data, 0.10, 25)] for data in original_images]
accuracy, op_imgs = test(weights, test_data)

# Print accuracy
print("Accuracy of the network is %f" % (accuracy * 100))

# Plot test result
plot_images(op_imgs, "Reconstructed Data", n_train_disp)
plt.show()
"""
