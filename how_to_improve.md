
# よりよいモデルを作成する方法

1. データをさらに集める
2. モデルを変更する
3. drive.pyを変更する



## 1. データをさらに集める


1.1. データを保存するディレクトリの作成
```
mkdir data_each_situation/base2
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

1.7 1.1に戻って同じことを繰り返したくさんデータを集める


## 2. モデルを変更する

`train_model.py`内の以下の部分を変更する。

```
# 元々こうなっている
MODEL_NAME = NVIDIA
# MODEL_NAME = RESNET50


# このように変更
# MODEL_NAME = NVIDIA
MODEL_NAME = RESNET50
```


## 3. drive.pyを変更する

///
