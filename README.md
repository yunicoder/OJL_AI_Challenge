# OJL_AI_Challenge

## Environment
```
pip install -r requirements.txt
```

## Usage

1. create data directory
```
# need to create once
mkdir data_each_situation

# need to create each data
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
python train_model -c [config_file_name]
```

5. autonomous drive using trained model
```
python drive.py model/[model_name].h5
```
