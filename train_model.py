import argparse
import h5py
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Dropout, MaxPooling2D
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from utils.resnet import resnet50_model
from keras.backend import tensorflow_backend as backend


# model name
NVIDIA = "nvidia"
RESNET50 = "resnet50"

MODEL_NAME = NVIDIA
# MODEL_NAME = RESNET50


def option_parser() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source_training_data_path',
                        required=True,
                        type=str,
                        help='path to source training data file')
    args = parser.parse_args()

    return args.source_training_data_path


def trainModelAndSave(model, inputs, outputs, epochs, batch_size):
    
    X_train, X_valid, y_train, y_valid = train_test_split(inputs, outputs, test_size=0.2, shuffle=True)
    #Setting model
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=1.0e-4))
    #Learning model
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs, verbose=1, validation_data=(X_valid, y_valid))
    #Saving model
    model.save("models/" + MODEL_NAME + ".h5")


#NVIDIA
def nvidia_model():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Conv2D(24,5,5, subsample=(2,2), activation='elu'))
    model.add(Conv2D(36,5,5, subsample=(2,2), activation='elu'))
    model.add(Conv2D(48,5,5, subsample=(2,2), activation='elu'))
    model.add(Conv2D(64,3,3, activation='elu'))
    model.add(Conv2D(64,3,3, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    return model


def main(source_training_data_path):
    if MODEL_NAME == NVIDIA:
        epochs = 10
        batch_size = 40
        model = nvidia_model()
    elif MODEL_NAME == RESNET50:  
        epochs = 1
        batch_size = 5
        model = resnet50_model()

    with h5py.File(source_training_data_path, 'r') as f:
        inputs = np.array(f['inputs'])
        outputs = np.array(f['outputs'])

    print('Training data:', inputs.shape)
    print('Training label:', outputs.shape)

    print("start " + MODEL_NAME + " model training...")
        
    #Training and saving model
    trainModelAndSave(model, inputs, outputs, epochs, batch_size)

    backend.clear_session()


if __name__ == '__main__':
    source_training_data_path = option_parser()
    main(source_training_data_path)
