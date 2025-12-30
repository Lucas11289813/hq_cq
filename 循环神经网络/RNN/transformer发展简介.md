# transformer发展简介

## 部分模型分析

![](md-img/transformer_2024-08-30-12-02-06.png)

![](md-img/transformer_2024-08-30-12-03-27.png)

![](md-img/transformer_2024-08-30-12-04-18.png)

## 开源大模型

![](md-img/transformer_2024-08-30-11-56-01.png)

![](md-img/transformer_2024-08-30-11-58-50.png)

模型参数量 1b = 10亿

## 名词解析

![](md-img/transformer_2024-08-30-12-06-11.png)

## 模型数据来源与差异

![](md-img/transformer_2024-08-30-12-08-02.png)

## 大模型发展的主要分支

![](md-img/transformer_2024-08-30-12-21-58.png)

GLM2 开始就借鉴了 LLama 采用了 Decoder-Only 结构

## 大公司发展

![](md-img/transformer_2024-08-30-12-30-42.png)

![](md-img/transformer_2024-08-30-12-31-11.png)

![](md-img/transformer_2024-08-30-12-35-51.png)

![](md-img/transformer_2024-08-30-12-36-28.png)

![](md-img/transformer_2024-08-30-12-40-40.png)

![](md-img/transformer_2024-08-30-12-42-44.png)

## encoder-decoder

![](md-img/transformer_2024-08-30-14-13-46.png)

输入和输出数量不同，且输出数量应该更多却没有得到正确数量输出时，称为**失步**

encoder-decoder 结构如下

![](md-img/transformer_2024-08-30-14-22-37.png)

训练方法有几种

![](md-img/transformer_2024-08-30-14-25-23.png)

上图描述的是 free-running mode，自由运行模式，其思想是拿预测的结果作为下一次的输入预测下一个字，这种方法模型很难训练或学到正确的内容

![](md-img/transformer_2024-08-30-14-28-09.png)

上图的训练方式叫 teacher-forcing mode，也就是教师强迫训练，我们用真实值 Ground True 作为输入训练模型

free-running mode 和 teacher-forcing mode 通常会一块儿使用，先用 teacher-forcing mode 强制教会模型基本信息，再用 free-running mode 让模型更具泛化性