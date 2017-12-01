# vqa
Visual Question Answering System

# How to run

## create environment

```bash
cd vqa
conda env create -f envs/environment.yml
```

## download data

```bash
cd vqa
mkdir data
cd data

## download necessary files
wget http://visualqa.org/data/abstract_v002/scene_img/scene_img_abstract_v002_train2015.zip
wget http://visualqa.org/data/abstract_v002/scene_img/scene_img_abstract_v002_val2015.zip
wget http://visualqa.org/data/abstract_v002/vqa/Questions_Train_abstract_v002.zip
wget http://visualqa.org/data/abstract_v002/vqa/Questions_Val_abstract_v002.zip
wget http://visualqa.org/data/abstract_v002/vqa/Annotations_Train_abstract_v002.zip
wget http://visualqa.org/data/abstract_v002/vqa/Annotations_Val_abstract_v002.zip

## unzip them

```

## train the network

```
source activate vqa_env
python train.py
```
