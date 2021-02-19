# AlexNet

from sklearn.pipeline import Pipeline
from tensorflow import keras
from transformers.cwt import Wavelet


# Function to create model, required for KerasClassifier
def create_model():
    model = keras.models.Sequential([
        keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu',
                            input_shape=(227, 227, 1)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=384, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='softmax')
    ])

    loss = 'binary_crossentropy'
    optimizer = 'adam'
    metrics = ['accuracy']
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def instantiate_auto_alexnet():

    alexnet = keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model, verbose=0)

    alexnet_pipe = Pipeline([
        ('Wavelet', Wavelet()),
        ('alexnet', alexnet),
    ])

    return alexnet_pipe
