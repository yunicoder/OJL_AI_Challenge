import os
import csv
import cv2
import h5py
import argparse
import numpy as np

from tqdm import tqdm


def option_parser() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir_name',
                        required=True,
                        type=str,
                        help='./data_each_situation/[THIS ARG]')
    args = parser.parse_args()

    return args.data_dir_name


def getrowsFromDrivingLogs(dataDir):
    rows = []
    with open(dataDir + '/driving_log.csv') as csvFile:
        reader = csv.reader(csvFile)            
        for row in reader:
            rows.append(row)
    return rows


def getImagesAndSteerings(data_dir, rows):
    def getImageArray3angle(imagePath, steering, images, steerings):
        originalImage = cv2.imread(imagePath.strip())
        image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
        images.append(image)
        steerings.append(steering)

    def getRelativeImagePath(ori_image_path, data_dir) -> str:
        relative_image_path = (data_dir
                               + '/IMG/'
                               + os.path.basename(ori_image_path))

        return relative_image_path


    images = []
    steerings = []
    
    for row in tqdm(rows):
        steering = float(row[3])

        ### 左右のカメラのステアリング測定値を調整
        parameter = 0.05  # このパラメータが調整用の値
        steering_left = steering + parameter
        steering_right = steering - parameter

        ### getImageArray3angle
        # center
        if row[0]:
            getImageArray3angle(getRelativeImagePath(row[0], data_dir),
                                steering,
                                images,
                                steerings)
        # left
        if row[1]:
            getImageArray3angle(getRelativeImagePath(row[1], data_dir),
                                steering_left,
                                images,
                                steerings)
        # right
        if row[2]:
            getImageArray3angle(getRelativeImagePath(row[2], data_dir),
                                steering_right,
                                images,
                                steerings)

    return (np.array(images), np.array(steerings))


if __name__ == '__main__':
    data_dir_name = option_parser()
    data_dir = 'data_each_situation/' + data_dir_name

    print('get csv data from Drivinglog.csv')
    rows = getrowsFromDrivingLogs(data_dir)
    print('preprocessing data...')
    inputs, outputs = getImagesAndSteerings(data_dir, rows)

    ### Create training data file (.h5)
    out_file_path = './training_data/' + data_dir_name + '.h5'
    with h5py.File(out_file_path, 'w') as f:
        f.create_dataset('inputs', data=inputs)
        f.create_dataset('outputs', data=outputs)
