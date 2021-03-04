# CNN

from sklearn.pipeline import Pipeline
from tensorflow import keras


# Function to create model, required for KerasClassifier
def create_model():

    model_m = keras.Sequential()
    model_m.add(keras.layers.Reshape((8192, 1), input_shape=(8192,1)))
    model_m.add(keras.layers.Conv1D(100, 10, activation='relu', input_shape=(8192, 1)))
    model_m.add(keras.layers.Conv1D(100, 10, activation='relu'))
    model_m.add(keras.layers.MaxPooling1D(3))
    model_m.add(keras.layers.Conv1D(160, 10, activation='relu'))
    model_m.add(keras.layers.Conv1D(160, 10, activation='relu'))
    model_m.add(keras.layers.GlobalAveragePooling1D())
    model_m.add(keras.layers.Dropout(0.5))
    model_m.add(keras.layers.Dense(1, activation='softmax'))
    print(model_m.summary())

    loss = 'binary_crossentropy'
    optimizer = 'adam'
    metrics = ['accuracy']
    model_m.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model_m

def instantiate_auto_cnn():

    cnn = keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model, verbose=0)

    cnn_pipe = Pipeline([
        ('cnn', cnn),
    ])

    return cnn_pipe
