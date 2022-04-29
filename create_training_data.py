import os
import cv2
import csv
from tqdm import tqdm
import h5py
import numpy as np

def path_leaf(path):
  head, tail = os.path.split(path)
  return tail


def getrowsFromDrivingLogs(dataPath):
    rows = []
    with open(dataPath + '/driving_log.csv') as csvFile:
        reader = csv.reader(csvFile)            
        for row in reader:
            row[0]=os.path.join(dataPath+ '/IMG', path_leaf(row[0]).strip())
            row[1]=os.path.join(dataPath+ '/IMG', path_leaf(row[1]).strip())
            row[2]=os.path.join(dataPath+ '/IMG', path_leaf(row[2]).strip())
            rows.append(row)
    return rows

def img_flip(image, steering):
    image = cv2.flip(image,1)
    steering = -steering
    return image, steering

def random_brightness(images):
    brightness_images = []
    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        brightness = .25 + np.random.uniform()
        image[:,:,2] = brightness*image[:,:,2]
        new_img = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        brightness_images.append(new_img)
    return brightness_images 

def flip(images,steerings):
    flipped_images = []
    flipped_steering_angles = []
    for original_image, steering_angle in zip(images,steerings):
        flipped_image, flipped_steering_angle = img_flip(original_image, steering_angle)
        flipped_images.append(flipped_image)
        flipped_steering_angles.append(flipped_steering_angle)
    images = np.concatenate((flipped_images, images))
    steerings = np.concatenate((flipped_steering_angles, steerings))
    return images, steerings

def getImageArray3angle(imagePath, steering, images, steerings):
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
        parameter = 0.15 
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
    inputs,outputs = flip(inputs,outputs)
    inputs = random_brightness(inputs)
    with h5py.File('./trainingData.h5', 'w') as f:
        f.create_dataset('inputs', data=inputs)
        f.create_dataset('outputs', data=outputs)
