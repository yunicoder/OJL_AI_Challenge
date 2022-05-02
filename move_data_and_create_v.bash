#!/bin/bash


### initialize option variables
DATA_DIR_NAME=""
SOURCE_DIR="$(dirname $0)/"


### parse command options
OPT=`getopt -o d: -l data_dir_name: -- "$@"`


if [ $? != 0 ] ; then
    echo "[Error] Option parsing processing is failed." 1>&2
    show_usage
    exit 1
fi

eval set -- "$OPT"

while true
do
    case $1 in
    -n | --data_dir_name)
        DATA_DIR_NAME="$2"
        shift 2
        ;;
    --)
        shift
        break
        ;;
    esac
done


### move data
mv /mnt/c/Users/atsushi/Downloads/${DATA_DIR_NAME}.zip /mnt/c/Users/atsushi/Documents/workspace/env/ACS/SaiI/data_each_situation/${DATA_DIR_NAME}.zip
unzip ${SOURCE_DIR}/data_each_situation/${DATA_DIR_NAME}.zip -d ${SOURCE_DIR}/data_each_situation/
rm -r ${SOURCE_DIR}/data_each_situation/${DATA_DIR_NAME}.zip
rm -r ${SOURCE_DIR}/data_each_situation/${DATA_DIR_NAME}/.ipynb_checkpoints

### create video
python3 ${SOURCE_DIR}/video.py data_each_situation/${DATA_DIR_NAME}/IMG
mv ${SOURCE_DIR}/data_each_situation/${DATA_DIR_NAME}/IMG.mp4 data_each_situation/${DATA_DIR_NAME}/${DATA_DIR_NAME}.mp4
