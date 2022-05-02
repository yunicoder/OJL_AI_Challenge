#!/bin/bash


### initialize option variables
SOURCE_DIR="$(dirname $0)/"
DOWNLOADS="/mnt/c/Users/atsushi/Downloads/"
WORKSPACE="/mnt/c/Users/atsushi/Documents/workspace/env/ACS/SaiI/"


### move data
ZIP_FILES=$( find ${DOWNLOADS} -name "*.zip" )
for zip_file_path in ${ZIP_FILES}
do
    zip_file_name=$( basename ${zip_file_path} )
    mv ${zip_file_path} ${WORKSPACE}/data_each_situation/
    unzip ${SOURCE_DIR}/data_each_situation/${zip_file_name} -d ${SOURCE_DIR}/data_each_situation/
    rm -r ${SOURCE_DIR}/data_each_situation/${zip_file_name}
    rm -r ${SOURCE_DIR}/data_each_situation/${zip_file_name}/.ipynb_checkpoints
done
