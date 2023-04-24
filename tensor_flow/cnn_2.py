from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

X = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
])

y = np.array([
    [0],
    [0],
    [0],
    [1],
    [1],
    [1],
    [1],
    [1],
])

n_features = 3

inputs = keras.layers.Input(name="input", shape=(n_features,))
### hidden layer 1
h1 = keras.layers.Dense(name="h1", units=int(round((n_features+1)/2)), activation='relu')(inputs)
h1 = keras.layers.Dropout(name="drop1", rate=0.2)(h1)
### hidden layer 2
h2 = keras.layers.Dense(name="h2", units=int(round((n_features+1)/4)), activation='relu')(h1)
h2 = keras.layers.Dropout(name="drop2", rate=0.2)(h2)
### layer output
outputs = keras.layers.Dense(name="output", units=1, activation='sigmoid')(h2)
model = keras.models.Model(inputs=inputs, outputs=outputs, name="DeepNN")
# model.summary()


model.compile(
    optimizer=keras.optimizers.RMSprop(),  # Optimizer
    # Minimize loss:
    loss=keras.losses.SparseCategoricalCrossentropy(),
    # Monitor metrics:
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

training = model.fit(x=X, y=y, batch_size=32, epochs=100, shuffle=True, verbose=0, validation_split=0.3)

data = np.array([
    [1, 1, 1],

])

# res = model.predict(data)
# print(res.shape)