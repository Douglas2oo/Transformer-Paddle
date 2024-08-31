# Transformer-Paddle_SiameseNetwork

2 Models (Transformer & Paddle_SiameseNetwork) for 2 texts Correlation Classification

- [Transformer-Paddle\_SiameseNetwork](#transformer-paddle_siamesenetwork)
  - [Introduction](#introduction)
  - [Overview](#overview)
    - [data](#data)
    - [.gitattributes](#gitattributes)
    - [Transformer Model](#transformeripynb)
    - [Paddle_Siamese](#paddle_siameseipynb)
    - [Paddle_Siamese with Parameter Optimization](#paddle_siamese_para_optipynb)
    


## Introduction

Given the title of a fake news article A and the title of a coming news article Bparticipants are asked to classify B into one of the three categories.

* agreed: B talks about the same fake news as A
* disagreed: B refutes the fake news in A
* unrelated: B is unrelated to A

Training Language: English.

## Overview

### data:
* train.csv (tracked by .gitattributes)
* test.csv
* solution.csv
* train_cleaned.csv: Preprocessed training data
* test_cleaned.csv: Preprocessed test data merging solution data

### .gitattributes:
tracking:
```
1. data/train.csv
2. transformer_weights/epoch_3_valid_acc_68.9_transformer_weights.bin
```

### transformer.ipynb:
Data Preprocessing & Training Process
#### Structure:
```
|-- 数据导入
|-- 数据预处理
        |-- 删去中文列
        |-- 合并测试集
        |-- 文本清理
        |-- 标签编码
        |-- 已预处理数据导出
        |-- 已预处理数据导入
        |-- 随机抽样缩小原数据集
|-- 文本向量化
        |-- 加载数据集（字典化）
        |-- 向量化处理
|-- 训练模型
        |-- 构建模型
        |-- 输出结构
        |-- 训练步骤数
        |-- 训练，测试准备
            |-- 训练循环
            |-- 测试循环
            |-- 加权准确率函数
        |-- 导入原最佳模型权重
        |-- 训练执行
        |-- 加权得分计算
```
### paddle_Siamese.ipynb:
Training Process
#### Structure:
```
|-- 导入已预处理数据
        |-- 随机抽样缩小原数据集
|-- 文本向量化
        |-- 加载数据集（字典化）
        |-- 向量化处理
|-- 模型训练准备
        |-- 定义模型
        |-- 训练循环
        |-- 测试循环
        |-- 加权准确率函数
|-- 训练模型
        |-- 向量化
        |-- 模型初始化
        |-- 输出结构
        |-- 训练步骤次数
        |-- 导入最佳模型权重
        |-- 训练执行
        |-- 加权准确率计算
```

### paddle_Siamese_para_opt.ipynb:
Parameter Optimization & Training Process
#### Structure:
```
|-- 导入已预处理数据
        |-- 随机抽样缩小原数据集
|-- 文本向量化
        |-- 加载数据集（字典化）
        |-- 向量化处理
|-- 模型训练准备
        |-- 定义模型
        |-- 训练循环
        |-- 测试循环
        |-- 加权准确率函数
|-- 超参数调参
            |-- 设置调参数据集
            |-- 定义搜索空间
            |-- 调参执行
|-- 训练模型
        |-- 初始化
                |-- 超参数调参结果输入
                |-- 向量化
                |-- 模型初始化
                |-- 输出结构
                |-- 训练步骤次数
                |-- 导入最佳模型权重
        |-- 训练执行
        |-- 加权准确率计算
```

