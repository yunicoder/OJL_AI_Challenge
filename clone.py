import sys
import csv
from tqdm import tqdm
import cv2
import datetime
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


def getrowsFromDrivingLogs(dataPath):
    rows = []
    with open(dataPath + '/driving_log.csv') as csvFile:
        reader = csv.reader(csvFile)            
        for row in reader:
            rows.append(row)
    return rows

def getImageArray3angle(imagePath, steering, images, steerings):
    originalImage = cv2.imread(imagePath.strip())
    image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
    if MODEL_NAME == RESNET50:
      image = cv2.resize(image, dsize=(480,240)) # for resnet50
    images.append(image)
    steerings.append(steering)
    
def getImagesAndSteerings(rows):
    
    images = []
    steerings = []
    
    for row in tqdm(rows):
        #angle
        steering = float(row[3])
        
        # 左右のカメラのステアリング測定値を調整します
        parameter = 0.15 
        # このパラメータが調整用の値です。
        # 左のカメラはステアリング角度が実際よりも低めに記録されているので、少し値を足してやります。右のカメラはその逆です。
        steering_left = steering + parameter
        steering_right = steering - parameter
        
        #center
        getImageArray3angle(row[0], steering, images, steerings)
        #left
        getImageArray3angle(row[1], steering_left, images, steerings)
        #right
        getImageArray3angle(row[2], steering_right, images, steerings)
        
    
    return (np.array(images), np.array(steerings))

def trainModelAndSave(model, inputs, outputs, epochs, batch_size):
    
    X_train, X_valid, y_train, y_valid = train_test_split(inputs, outputs, test_size=0.2, shuffle=True)
    #Setting model
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=1.0e-4))
    #Learning model
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs, verbose=1, validation_data=(X_valid, y_valid))
    #Saving model
    model.save("models/" + MODEL_NAME + ".h5")
    
#essential network
def nn_model():
    model = Sequential()
    model.add(Flatten(input_shape=(160, 320, 3)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1))
    return model

#convolutional network
def cnn_model():
    model = Sequential()
    model.add(Conv2D(32, 3, 3, subsample=(2,2), activation='relu', input_shape=(160, 320, 3)))
    model.add(Conv2D(64, 3, 3, subsample=(2,2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1))
    return model

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

if __name__ == "__main__":

    if MODEL_NAME == NVIDIA:
        epochs = 10
        batch_size = 40
        model = nvidia_model()
    elif MODEL_NAME == RESNET50:  
        epochs = 1
        batch_size = 5
        model = resnet50_model()

    
    #Make sure to set batch size within 40
    is_dataset = False
    # is_dataset = False
    
    #When making "is_dataset" True, saving preprocessed datasets
    #When making "is_dataset" False, using preprocessed and saved datasets. Shortens time because preprocessing is not required.It must be preprocessed once.
    if is_dataset:
        print('get csv data from Drivinglog.csv')
        rows = getrowsFromDrivingLogs('data')
        print('preprocessing data...')
        inputs, outputs = getImagesAndSteerings(rows)
        
        with h5py.File('./trainingData.h5', 'w') as f:
            f.create_dataset('inputs', data=inputs)
            f.create_dataset('outputs', data=outputs)
    
    else:
        with h5py.File('./trainingData.h5', 'r') as f:
            inputs = np.array(f['inputs'])
            outputs = np.array(f['outputs'])

    print('Training data:', inputs.shape)
    print('Training label:', outputs.shape)

    print("start " + MODEL_NAME + " model training...")
        
    #Training and saving model
    trainModelAndSave(model, inputs, outputs, epochs, batch_size)

    backend.clear_session()
