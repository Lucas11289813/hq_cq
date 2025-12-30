import json
import os
import traceback

import torch


# model: 要保存的模型
# save_dir: 保存的目录路径
# epoch: 训练的轮数
# loss: 损失值
# lr: 学习率
def save(model, save_dir, epoch, loss, lr):
    model_name = model.__class__.__name__
    meta_path = os.path.join(save_dir, f'{model_name}.meta.json')

    try:
        with open(meta_path, 'r') as file:
            meta = json.load(file)
    except:
        print('未找到元数据，初始化 meta')
        meta = {
            'epoch': [],
            'loss': [],
            'lr': [],
            'avg_loss': 0
        }

    meta['epoch'].append(epoch)
    meta['loss'].append(loss)
    meta['lr'].append(lr)
    meta['avg_loss'] = sum(meta['loss']) / len(meta['loss'])

    model_path = os.path.join(save_dir, f'{model_name}.{epoch}.pth')

    try:
        torch.save(model.state_dict(), model_path)
        with open(meta_path, 'w') as file:
            json.dump(meta, file)
        print('保存成功')
    except:
        print('保存失败')
        traceback.print_exc()


# 加载模型
# model_class: 模型类
# save_dir: 保存的目录路径
# epoch: 训练的轮数，为 None 时则加载最后一个训练结果
# construct_param: 构造模型的参数字典
def load(model_class, save_dir, epoch=None, construct_param=None):
    model_name = model_class.__name__
    meta_path = os.path.join(save_dir, f'{model_name}.meta.json')

    model = model_class() if construct_param is None else model_class(**construct_param)

    try:
        with open(meta_path, 'r') as file:
            meta = json.load(file)
        epoch = epoch if epoch is not None else meta['epoch'][-1]
        model_path = os.path.join(save_dir, f'{model_name}.{epoch}.pth')
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        print('加载成功')
    except:
        print('未找到元数据，初始化模型')
        meta = {
            'epoch': [],
            'loss': [],
            'lr': [],
            'avg_loss': 0
        }

    return model, meta
