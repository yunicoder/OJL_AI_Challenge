# OJL_AI_Challenge

## Environment
```
pip install -r requirements.txt
```

## Usage

1. create data directory
```
# create once
mkdir data_each_situation

# create for each data
mkdir data_each_situation/[data_name]
```

2. create data by running simulator

Data should store in data_each_situation/[data_name]

3. convert data to h5 
```
mkdir training_data
python create_training_data.py -d [data_name]
```

4. train model
```
mkdir models
python train_model.py -c [config_file_name]
```

5. autonomous drive using trained model
```
python drive.py models/[model_name].h5
```


### for B4
1. create data directory
```
mkdir data_each_situation
mkdir data_each_situation/base
```

2. create data by running simulator

Simulation data store in data_each_situation/base

3. convert data to h5 
```
mkdir training_data
python create_training_data.py -d base
```

4. edit `training_config/base.yaml`
```
use_data:
  base:
    flip: False

must_train_data:
  only_one_image:
    flip: False
```


5. train model
```
mkdir models
python train_model.py -c base
```

6. autonomous drive using trained model
```
python drive.py models/nvidia.h5
```
