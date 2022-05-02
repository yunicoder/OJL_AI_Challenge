#!/bin/bash


### initialize option variables
NEW_NAME=""
SOURCE_DIR="$(dirname $0)/"


### parse command options
OPT=`getopt -o n: -l new_name: -- "$@"`


if [ $? != 0 ] ; then
    echo "[Error] Option parsing processing is failed." 1>&2
    show_usage
    exit 1
fi

eval set -- "$OPT"

while true
do
    case $1 in
    -n | --new_name)
        NEW_NAME="$2"
        shift 2
        ;;
    --)
        shift
        break
        ;;
    esac
done


### run video.py
python3 ${SOURCE_DIR}/video.py data_each_situation/data_temp/IMG


### remove & rename
rm -r ${SOURCE_DIR}/data_each_situation/data_temp.zip
mv ${SOURCE_DIR}/data_each_situation/data_temp data_each_situation/${NEW_NAME}
mv ${SOURCE_DIR}/data_each_situation/${NEW_NAME}/IMG.mp4 data_each_situation/${NEW_NAME}/${NEW_NAME}.mp4
