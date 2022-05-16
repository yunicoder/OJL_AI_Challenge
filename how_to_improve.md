
# よりよいモデルを作成する方法

1. データをさらに集める
2. モデルを変更する
3. drive.pyを変更する



## 1. データをさらに集める


1.1. データを保存するディレクトリの作成
```
mkdir data_each_situation/base
```

1.2. シミュレータを起動して、base2というディレクトリに新しいデータを集める

1.3. base2のデータをh5に変換する
```
python create_training_data.py -d base2
```

1.4. `training_config/base.yaml`を以下のように変更する

```
use_data:
  base:
    flip: False
  base2:          # ここを追加
    flip: False

must_train_data:
  only_one_image:
    flip: False
```

1.5. base.yamlに書いてあるデータを使ってモデルを学習させる
```
python train_model.py -c base
```

1.6. シミュレータ上でモデルを走らせる

```
# モデルを実行する
python drive.py models/nvidia.h5

# 実行出来たらシミュレータを起動する
```


## 2. モデルを変更する

///

## 3. drive.pyを変更する

///
