#!/bin/bash

DATA_DIRS=(bridge_recovery_from_left2 bridge_recovery_from_right2 bridge_to_curve2_from_in_to_center bridge_to_curve2_from_out_to_center)


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
