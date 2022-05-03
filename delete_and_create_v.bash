#!/bin/bash

DATA_DIRS=(curve2_always_center curve2_always_in curve2_always_out curve3_always_center curve3_always_in curve3_always_out curve4_always_center curve4_always_in curve4_always_out)


### delete unused image
for data_dir in ${DATA_DIRS[@]}
do
    python3 delete_unused_image.py --data_path ./data_each_situation/${data_dir}
done


### create video
for data_dir in ${DATA_DIRS[@]}
do
    python3 video.py ${data_dir}
done
