#!/bin/bash


### initialize option variables
SOURCE_DIR="$(dirname $0)/"


### move data
ZIP_FILES=$( find ${SOURCE_DIR} -name "*.zip" )
for zip_file_name in ${ZIP_FILES}
do
    mv /mnt/c/Users/atsushi/Downloads/${zip_file_name}.zip /mnt/c/Users/atsushi/Documents/workspace/env/ACS/SaiI/data_each_situation/${zip_file_name}.zip
    unzip ${SOURCE_DIR}/data_each_situation/${zip_file_name}.zip -d ${SOURCE_DIR}/data_each_situation/
    rm -r ${SOURCE_DIR}/data_each_situation/${zip_file_name}.zip
    rm -r ${SOURCE_DIR}/data_each_situation/${zip_file_name}/.ipynb_checkpoints
done
