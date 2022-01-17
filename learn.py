# načtení knihovny a dat pro ML
import tensorflow as tf #pytorch
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# Průzkum stažené datové sady

print(x_train[0])
print()
print(y_train[0])

import matplotlib.pyplot as plt
plt.imshow(x_train[456], cmap="gray")
print(y_train[456])


# normalizace dat

x_train = x_train / 255.0
x_test = x_test / 255.0
print(x_test[0])


# vytvoření neuronové sítě

vstupni = tf.keras.layers.Flatten(input_shape=(28, 28))
skryte = tf.keras.layers.Dense(128, activation="relu")
zahazujici = tf.keras.layers.Dropout(0.1)
vystupni = tf.keras.layers.Dense(10)

neuronova_sit = tf.keras.model.Sequential([vstupni, skryte, zahazujici, vystupni])
