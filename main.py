from keras import layers, models, optimizers
import numpy as np
import matplotlib.pyplot as plt


X_train = np.array([
    [0, 1],
    [1, 0],
    [1, 1],
    [0, 0],
])

y_train = np.array([
    0,
    0,
    1,
    1,
])


model = models.Sequential()
model.add(layers.Dense(8, activation='tanh'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=1000, batch_size=1, verbose=0)
model.summary()

classes = model.predict(X_train)

plt.plot(history.history['loss'])
plt.show()

for point, label in zip(X_train, classes):
    color_code = np.where(label[0] > 0.5, 1, 0)
    colors = ['r', 'b']
    plt.scatter(point[0], point[1], c=colors[color_code])

plt.show()