import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np
import matplotlib.pyplot as plt

def model(X_train, X_test, y_train, y_test, dropout_rate=0.1, nb_epochs=10):
    X_shape = X_train[0].shape
    nb_classes = len(np.unique(y_train))

    my_CNN = models.Sequential()
    my_CNN.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=X_shape))
    my_CNN.add(layers.MaxPooling2D((2, 2)))
    my_CNN.add(layers.Flatten())
    my_CNN.add(layers.Dense(32, activation="relu"))
    my_CNN.add(layers.Dropout(rate=dropout_rate))
    my_CNN.add(layers.Dense(nb_classes, activation="softmax"))

    cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    my_CNN.compile(optimizer='adam',
                loss=cross_entropy,
                metrics=['accuracy'])
    
    tf_history = my_CNN.fit(X_train, y_train, epochs=nb_epochs, validation_data=(X_test, y_test))
    tf_accuracy = my_CNN.evaluate(X_test,  y_test, verbose=2)

    return tf_accuracy, tf_history

def plot_loss(tf_accuracy, tf_history, nb_epochs):
    print(f"Test accuracy for CNN is: {round(tf_accuracy[1], 3)}")
    print("----"*20)

    epochs = range(nb_epochs)

    plt.figure(figsize=(8,5))
    plt.plot(tf_history.history["loss"], label="train_loss")
    plt.plot(tf_history.history["val_loss"], label = "test_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(epochs)
    plt.xlim([0, int(max(epochs))])
    plt.ylim([0, tf_accuracy[0]])
    plt.legend(loc="upper right")
    plt.show()