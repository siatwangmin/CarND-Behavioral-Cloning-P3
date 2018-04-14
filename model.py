import tensorflow as tf
from keras.layers import Dense, Flatten, Lambda, Activation, MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.optimizers import Adam

import utils
import matplotlib.pyplot as plt

tf.python.control_flow_ops = tf

number_of_epochs = 30
number_of_samples_per_epoch = 20032
number_of_validation_samples = 6400
learning_rate = 1e-4
activation_relu = 'relu'

# The model is based on NVIDIA's "End to End Learning for Self-Driving Cars" paper
model = Sequential()

# Normallize the data
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(64, 64, 3)))

# starts with five convolutional and maxpooling layers
# input (64*64*3), output (32*32*24)
model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation(activation_relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

# input (32*32*24), output (16*16*36)
model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation(activation_relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

# input (16*16*36), output (8*8*48)
model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation(activation_relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

# input (8*8*48), output (8*8*64)
model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation(activation_relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

# input (8*8r*64), output (8*8*64)
model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation(activation_relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

#output 4096
model.add(Flatten())

# Next, five fully connected layers
model.add(Dense(1164))
model.add(Activation(activation_relu))

model.add(Dense(100))
model.add(Activation(activation_relu))

model.add(Dense(50))
model.add(Activation(activation_relu))

model.add(Dense(10))
model.add(Activation(activation_relu))

model.add(Dense(1))

model.summary()

model.compile(optimizer=Adam(learning_rate), loss="mse", )

# create two generators for training and validation
train_gen = utils.generate_next_batch()
validation_gen = utils.generate_next_batch()

history_object = model.fit_generator(train_gen,
                              samples_per_epoch=number_of_samples_per_epoch,
                              nb_epoch=number_of_epochs,
                              validation_data=validation_gen,
                              nb_val_samples=number_of_validation_samples,
                              verbose=1)
# after trainning we save our model and weights
model.save('model.h5')
# Finally we plot the loss and validation loss
print(history_object.history.keys)
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()