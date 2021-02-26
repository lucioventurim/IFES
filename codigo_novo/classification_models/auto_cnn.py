# CNN

from sklearn.pipeline import Pipeline
from tensorflow import keras


# Function to create model, required for KerasClassifier
def create_model():
    model = keras.models.Sequential([
        keras.layers.InputLayer(input_shape=(664, 8192, 1)),
        keras.layers.Conv1D(32, 32, padding='valid'),
        keras.layers.Activation('relu'),
        keras.layers.GlobalMaxPooling1D(),
        #keras.layers.MaxPooling1D(pool_size=8),
        keras.layers.Conv1D(32, 32, padding='valid'),
        keras.layers.Activation('relu'),
        keras.layers.GlobalMaxPooling1D(),
        #keras.layers.MaxPooling1D(pool_size=8),
        #keras.layers.Flatten(),
        keras.layers.Dense(64),
        keras.layers.Activation('relu'),
        keras.layers.Dropout(0.25),
        keras.layers.Dense(3),
        keras.layers.Activation('softmax')
    ])

    loss = 'binary_crossentropy'
    optimizer = 'adam'
    metrics = ['accuracy']
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def instantiate_auto_cnn():

    cnn = keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model, verbose=0)

    cnn_pipe = Pipeline([
        ('cnn', cnn),
    ])

    return cnn_pipe
