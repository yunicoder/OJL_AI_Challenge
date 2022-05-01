import os
import cv2
import csv
from tqdm import tqdm
import cv2
import h5py
import numpy as np


def getrowsFromDrivingLogs(dataPath):
    rows = []
    with open(dataPath + '/driving_log.csv') as csvFile:
        reader = csv.reader(csvFile)            
        for row in reader:
            rows.append(row)
    return rows


def getImageArray3angle(imagePath, steering, images, steerings):
    imagePath = './data/IMG/' + os.path.basename(imagePath)
    originalImage = cv2.imread(imagePath.strip())
    image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
    images.append(image)
    steerings.append(steering)


def getImagesAndSteerings(rows):
    
    images = []
    steerings = []
    
    for row in tqdm(rows):
        #angle
        steering = float(row[3])
        
        # 左右のカメラのステアリング測定値を調整します
        parameter = 0.05
        # このパラメータが調整用の値です。
        # 左のカメラはステアリング角度が実際よりも低めに記録されているので、少し値を足してやります。右のカメラはその逆です。
        steering_left = steering + parameter
        steering_right = steering - parameter
        
        #center
        if row[0]:
            getImageArray3angle(row[0], steering, images, steerings)
        #left
        if row[1]:
            getImageArray3angle(row[1], steering_left, images, steerings)
        #right
        if row[2]:
            getImageArray3angle(row[2], steering_right, images, steerings)
        
    
    return (np.array(images), np.array(steerings))


if __name__ == '__main__':
    print('get csv data from Drivinglog.csv')
    rows = getrowsFromDrivingLogs('data')
    print('preprocessing data...')
    inputs, outputs = getImagesAndSteerings(rows)
    
    with h5py.File('./trainingData.h5', 'w') as f:
        f.create_dataset('inputs', data=inputs)
        f.create_dataset('outputs', data=outputs)
