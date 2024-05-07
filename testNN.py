import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical, set_random_seed

model = keras.Sequential(
    [
        keras.Input(shape=(X_train.shape[-1],)),
        # A optimiser
        layers.Dense(10, activation=relu), 
        layers.Dense(4, activation="softmax"),
    ]
)
model.summary()

model.compile(optimizer=keras.optimizers.Adam(0.01), # learning rate a optimiser
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'],)

history=model.fit(X_train, y_train, epochs=100, validation_split = 0.1, batch_size=32)

print(model.summary())

import matplotlib.pyplot as plt
plt.plot(history.history['val_loss'],color='blue')
plt.plot(history.history['loss'],color='red')
plt.show()

y_pred = model.predict(X_test)


# save best model over epochs: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint