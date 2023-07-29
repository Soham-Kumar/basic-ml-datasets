from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# Normalizing the data
mean = train_data.mean(axis=0) 
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

# Model definition
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', 
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    # No activation function in the last layer
    model.add(layers.Dense(1))
    # Mean Squared Error (MSE) is the same as the loss function
    # used in the previous example
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

# K-fold validation
k = 4
num_val_samples = len(train_data) // k
num_epochs = 500
all_scores = []
for i in range(k):
    print('Processing fold #', i)
    # Preparing the validation data: data for partition #k
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples] 
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    # Preparing the training data: data from all other partitions
    partial_train_data = np.concatenate( 
        [train_data[:i * num_val_samples], 
         train_data[(i + 1) * num_val_samples:]], 
        axis=0)
    partial_train_targets = np.concatenate( 
        [train_targets[:i * num_val_samples], 
         train_targets[(i + 1) * num_val_samples:]], 
        axis=0)
    # Building the Keras model (already compiled)
    model = build_model()
    # Training the model (in silent mode, verbose=0)
    history = model.fit(partial_train_data, partial_train_targets, 
              epochs=num_epochs, batch_size=1, verbose=0)
    # Evaluating the model on the validation data
    mae_history = history.history['mae']
    all_scores.append(mae_history)

# Building the history of successive mean K-fold validation scores
average_mae_history = [np.mean([x[i] for x in all_scores]) for i in range(num_epochs)]
print(average_mae_history)

# Plotting validation scores
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()