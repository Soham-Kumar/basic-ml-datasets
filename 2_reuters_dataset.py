from keras.datasets import reuters
from keras.utils.np_utils import to_categorical
import numpy as np
from keras import models
from keras import layers
import matplotlib.pyplot as plt


(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

#Preparing data
def vectorize(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):  # i is the index, sequence is the value
        results[i, sequence] = 1.0  # set specific indices of results[i] to 1s
    return results

x_train = vectorize(train_data)
x_test = vectorize(test_data)

# def one_hot_encode(labels, dimension=46): #46 is the number of classes
#     results = np.zeros((len(labels), dimension))
#     for i, label in enumerate(labels):
#         results[i, label] = 1.0
#     return results

# one_hot_train_labels = one_hot_encode(train_labels)
# one_hot_test_labels = one_hot_encode(test_labels)

one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

# Building the network
model = models.Sequential()
model.add(layers.Dense(64, activation="relu", input_shape=(10000,)))  # 16 hidden units
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(46, activation="softmax"))  # softmax activation to output a probability distribution over 46 different output classes

model.compile(optimizer="rmsprop",
            loss="categorical_crossentropy",
            metrics=["accuracy"])

# Setting aside a validation set
x_val = x_train[:1000]  # first 1000 samples for validation
partial_x_train = x_train[1000:]  # remaining 7982 samples for training
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

# Training your model
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=12,
                    batch_size=512,
                    validation_data=(x_val, y_val))

# # Plotting the training and validation loss
# loss = history.history["loss"]
# val_loss = history.history["val_loss"]
# epochs = range(1, len(loss) + 1)

# plt.plot(epochs, loss, "bo", label="Training loss")  # "bo" is for "blue dot"
# plt.plot(epochs, val_loss, "b", label="Validation loss")  # "b" is for "solid blue line"
# plt.title("Training and validation loss")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()

# # Plotting the training and validation accuracy
# plt.clf()  # clears the figure
# acc = history.history["accuracy"]
# val_acc = history.history["val_acc"]

# plt.plot(epochs, acc, "bo", label="Training acc")  # "bo" is for "blue dot"
# plt.plot(epochs, val_acc, "b", label="Validation acc")  # "b" is for "solid blue line"
# plt.title("Training and validation accuracy")
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.show()

results = model.evaluate(x_test, one_hot_test_labels)
prediction = np.argmax(model.predict(x_test)[0])

# If we don't want to use one-hot encoding, we can use sparse_categorical_crossentropy as the loss function
# while using integral tensors. This avoids the need for converting labels to one-hot encoding.
