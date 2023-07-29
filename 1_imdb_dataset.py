from keras.datasets import imdb
import numpy as np

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# num_words=10000 means you'll only keep the top 10,000 most frequently occurring words in the training data.

# word_index = imdb.get_word_index()
# reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[15]])
# print(decoded_review)


def vectorize(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):  # i is the index, sequence is the value
        results[i, sequence] = 1.0  # set specific indices of results[i] to 1s
    return results


x_train = vectorize(train_data)
x_test = vectorize(test_data)

y_train = np.asarray(train_labels).astype("float32")  # convert to numpy array
y_test = np.asarray(test_labels).astype("float32")

# Building the network
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation="relu", input_shape=(10000,)))  # 16 hidden units
model.add(layers.Dense(16, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))  # sigmoid activation to output a probability

model.compile(optimizer="rmsprop",
                loss="binary_crossentropy",
                metrics=["accuracy"])

# # Configuring the optimizer
# from keras import optimizers
# model.compile(optimizer=optimizers.RMSprop(lr=0.001),
#                 loss="binary_crossentropy",
#                 metrics=["accuracy"])

# # Using custom losses and metrics
# from keras import losses
# from keras import metrics
# model.compile(optimizer=optimizers.RMSprop(lr=0.001),
#                 loss=losses.binary_crossentropy,   
#                 metrics=[metrics.binary_accuracy])

# Setting aside a validation set
x_val = x_train[:10000]  # first 10,000 samples for validation
partial_x_train = x_train[10000:]  # remaining 15,000 samples for training
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# Training your model
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=4,
                    batch_size=512,
                    validation_data=(x_val, y_val))
    # The call to model.fit() returns a History object. This object has a member history,
    # which is a dictionary containing data about everything that happened during training.
    # The dictionary contains four entries: one per metric that was being monitored during training and during validation.

# Plotting the training and validation loss
import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]

epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, "bo", label="Training loss")  # "bo" is for "blue dot"
plt.plot(epochs, val_loss_values, "b", label="Validation loss")  # "b" is for "solid blue line"
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()


# Plotting the training and validation accuracy
plt.clf()  # clear figure
acc_values = history_dict["accuracy"]
val_acc_values = history_dict["val_accuracy"]

plt.plot(epochs, acc_values, "bo", label="Training acc")
plt.plot(epochs, val_acc_values, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.show()

print(model.evaluate(x_test, y_test))