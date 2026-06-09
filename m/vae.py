import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.losses import mse
from tensorflow.keras import backend as K
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

print(x_train.shape, x_test.shape)
input_dim = 784
latent_dim = 8

inputs = Input(shape=(input_dim,))

# -------- Encoder --------
h1 = Dense(128, activation='relu')(inputs)
h2 = Dense(64, activation='relu')(h1)

mu = Dense(latent_dim)(h2)
log_var = Dense(latent_dim)(h2)
def sampling(args):
    mu, log_var = args
    epsilon = K.random_normal(shape=K.shape(mu))
    return mu + K.exp(0.5 * log_var) * epsilon

z = Lambda(sampling)([mu, log_var])
d1 = Dense(64, activation='relu')(z)
d2 = Dense(128, activation='relu')(d1)
outputs = Dense(input_dim, activation='sigmoid')(d2)

vae = Model(inputs, outputs)
reconstruction_loss = mse(inputs, outputs) * input_dim
kl_loss = -0.5 * K.sum(1 + log_var - K.square(mu) - K.exp(log_var), axis=-1)
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.summary()
history = vae.fit(
    x_train,
    epochs=20,
    batch_size=256,
    validation_data=(x_test, None),
    verbose=1
)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('VAE Loss')
plt.xlabel('Epochs')    
plt.ylabel('Loss')
plt.legend()
plt.show()