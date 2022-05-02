import argparse
import h5py
import cv2
import yaml
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Dropout, MaxPooling2D
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List

from utils.resnet import resnet50_model
from keras.backend import tensorflow_backend as backend



# model name
NVIDIA = "nvidia"
RESNET50 = "resnet50"

# MODEL_NAME = NVIDIA
MODEL_NAME = RESNET50


def option_parser() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--train_cfg_name',
                        required=True,
                        type=str,
                        help='training_config/[This arg].yaml')
    args = parser.parse_args()

    return args.train_cfg_name


def trainModelAndSave(model, inputs, outputs, must_train_inputs, must_train_outputs, epochs, batch_size):
    is_validation = False

    if is_validation:
        X_train, X_valid, y_train, y_valid = train_test_split(inputs, outputs, test_size=0.2, shuffle=True)
        X_train = np.concatenate([X_train, must_train_inputs], axis=0)
        y_train = np.concatenate([y_train, must_train_outputs], axis=0)
    else:
        inputs = np.concatenate([inputs, must_train_inputs], axis=0)
        outputs = np.concatenate([outputs, must_train_outputs], axis=0)
    #Setting model
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=1.0e-4))
    #Learning model
    if is_validation:
        model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs, verbose=1, validation_data=(X_valid, y_valid))
    else:
        model.fit(inputs, outputs, batch_size=batch_size, nb_epoch=epochs, verbose=1)
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


def preprocessing(model_input_shape, inputs, outputs, flip):
        images = []
        steerings = []
        
        for input, output in zip(inputs, outputs):
            ### resize to fit model input
            image = cv2.resize(input, (model_input_shape[2], model_input_shape[1]), interpolation=cv2.INTER_LINEAR)
            images.append(image)
            steerings.append(output)

            ### flip
            if flip:
                image_flip = cv2.flip(image,1)
                steering_flip = -output
                images.append(image_flip)
                steerings.append(steering_flip)

        return np.array(images), np.array(steerings)


def prepare_data(train_cfg, model_input_shape):
    inputs = outputs = None
    for train_data_name, cfg in train_cfg.items():
        train_data_path = './training_data/' + train_data_name + '.h5'
        with h5py.File(train_data_path, 'r') as f:
            if(inputs == None):
                inputs, outputs = preprocessing(
                    model_input_shape,
                    np.array(f['inputs']),
                    np.array(f['outputs']),
                    **cfg
                )
            else:
                _inputs, _outputs = preprocessing(
                    model_input_shape,
                    np.array(f['inputs']),
                    np.array(f['outputs']),
                    **cfg
                )
                inputs = np.concatenate([inputs,_inputs], axis=0)
                outputs = np.concatenate([outputs, _outputs], axis=0)

    return inputs, outputs


def main(train_cfg_name):
    ### create model
    if MODEL_NAME == NVIDIA:
        epochs = 10
        batch_size = 40
        model = nvidia_model()
    elif MODEL_NAME == RESNET50:  
        epochs = 1
        batch_size = 5
        model = resnet50_model()

    ### load training config
    train_cfg_path = './training_config/' + train_cfg_name + '.yaml'
    with open(train_cfg_path, 'r') as f:
        train_cfg_ori = yaml.load(f)

    ### prepare data
    inputs, outputs = prepare_data(train_cfg_ori['use_data'], model.input_shape)
    must_train_inputs, must_train_outputs = prepare_data(train_cfg_ori['must_train_data'], model.input_shape)
    print('Training data: ', inputs.shape)
    print('Training label: ', outputs.shape)
    print('must train data count: ', must_train_inputs.shape[0])
    print('all train data count: ', inputs.shape[0]+must_train_inputs.shape[0])

    ### Training and saving model
    print("start " + MODEL_NAME + " model training...")
    trainModelAndSave(model, inputs, outputs, must_train_inputs, must_train_outputs, epochs, batch_size)

    backend.clear_session()


if __name__ == '__main__':
    train_cfg_name = option_parser()
    main(train_cfg_name)
