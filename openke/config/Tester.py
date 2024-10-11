# coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import sys
import datetime
import ctypes
import json
import numpy as np
from sklearn.metrics import roc_auc_score
import copy
from tqdm import tqdm

# 定义Tester类，用于模型测试
class Tester(object):
    # 初始化函数，接收模型、数据加载器和是否使用GPU的参数
    def __init__(self, model = None, data_loader = None, use_gpu = True):
        # 加载C++编写的Base.so动态链接库，用于高效计算
        base_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../release/Base.so"))
        self.lib = ctypes.cdll.LoadLibrary(base_file)

        # 定义lib库中C函数的参数类型，用于链接预测任务
        self.lib.testHead.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
        self.lib.testTail.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
        self.lib.test_link_prediction.argtypes = [ctypes.c_int64]

        # 定义从C函数中获取评估结果的方法，用于链接预测评估指标
        self.lib.getTestLinkMRR.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkMR.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit10.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit3.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit1.argtypes = [ctypes.c_int64]

        # 设置这些方法的返回值类型为float
        self.lib.getTestLinkMRR.restype = ctypes.c_float
        self.lib.getTestLinkMR.restype = ctypes.c_float
        self.lib.getTestLinkHit10.restype = ctypes.c_float
        self.lib.getTestLinkHit3.restype = ctypes.c_float
        self.lib.getTestLinkHit1.restype = ctypes.c_float

        # 保存模型和数据加载器，并根据是否使用GPU移动模型到GPU
        self.model = model
        self.data_loader = data_loader
        self.use_gpu = use_gpu

        # 如果使用GPU，将模型移到GPU
        if self.use_gpu:
            self.model.cuda()

    # 设置新的模型
    def set_model(self, model):
        self.model = model

    # 设置新的数据加载器
    def set_data_loader(self, data_loader):
        self.data_loader = data_loader

    # 设置是否使用GPU，并在需要时将模型移到GPU
    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu
        if self.use_gpu and self.model != None:
            self.model.cuda()

    # 将numpy数组转换为PyTorch的Variable，并根据是否使用GPU移动到相应设备
    def to_var(self, x, use_gpu):
        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))

    # 测试单个批次的数据，调用模型的预测函数
    def test_one_step(self, data):
        return self.model.predict({
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),  # 转换头实体的批次数据
            'batch_t': self.to_var(data['batch_t'], self.use_gpu),  # 转换尾实体的批次数据
            'batch_r': self.to_var(data['batch_r'], self.use_gpu),  # 转换关系的批次数据
            'mode': data['mode']  # 模式（头预测或尾预测）
        })

    # 执行链接预测测试
    def run_link_prediction(self, type_constrain = False):
        self.lib.initTest()  # 初始化测试环境（由C++库提供）
        self.data_loader.set_sampling_mode('link')  # 设置数据加载器的采样模式为'link'模式
        # 设置类型约束标志，如果使用类型约束则为1，否则为0
        if type_constrain:
            type_constrain = 1
        else:
            type_constrain = 0

        # 使用tqdm显示进度条，遍历测试数据集
        training_range = tqdm(self.data_loader)
        for index, [data_head, data_tail] in enumerate(training_range):
            # 对头实体进行预测
            score = self.test_one_step(data_head)
            # 调用C++库的方法对头实体进行测试并记录结果
            self.lib.testHead(score.__array_interface__["data"][0], index, type_constrain)

            # 对尾实体进行预测
            score = self.test_one_step(data_tail)
            # 调用C++库的方法对尾实体进行测试并记录结果
            self.lib.testTail(score.__array_interface__["data"][0], index, type_constrain)

        # 调用C++库的方法完成链接预测的测试
        self.lib.test_link_prediction(type_constrain)

        # 获取测试指标：MRR（Mean Reciprocal Rank）、MR（Mean Rank）、Hit@10、Hit@3、Hit@1
        mrr = self.lib.getTestLinkMRR(type_constrain)
        mr = self.lib.getTestLinkMR(type_constrain)
        hit10 = self.lib.getTestLinkHit10(type_constrain)
        hit3 = self.lib.getTestLinkHit3(type_constrain)
        hit1 = self.lib.getTestLinkHit1(type_constrain)
        print (hit10)  # 输出Hit@10的值
        return mrr, mr, hit10, hit3, hit1  # 返回测试结果

    # 计算最佳阈值，用于三元组分类任务
    def get_best_threshlod(self, score, ans):
        # 将答案和预测得分组合为一个数组并按得分排序
        res = np.concatenate([ans.reshape(-1,1), score.reshape(-1,1)], axis = -1)
        order = np.argsort(score)
        res = res[order]

        # 计算所有正例和负例的总数
        total_all = (float)(len(score))
        total_current = 0.0
        total_true = np.sum(ans)
        total_false = total_all - total_true

        # 初始化最佳阈值和最大准确率
        res_mx = 0.0
        threshlod = None

        # 遍历排序后的得分，寻找最佳阈值
        for index, [ans, score] in enumerate(res):
            if ans == 1:
                total_current += 1.0
            res_current = (2 * total_current + total_false - index - 1) / total_all
            if res_current > res_mx:
                res_mx = res_current
                threshlod = score
        return threshlod, res_mx  # 返回最佳阈值及对应的准确率

    # 执行三元组分类任务
    def run_triple_classification(self, threshlod = None):
        self.lib.initTest()  # 初始化测试环境
        self.data_loader.set_sampling_mode('classification')  # 设置数据加载器的采样模式为'classification'
        score = []
        ans = []
        training_range = tqdm(self.data_loader)
        # 遍历数据集，分别处理正例和负例
        for index, [pos_ins, neg_ins] in enumerate(training_range):
            res_pos = self.test_one_step(pos_ins)  # 对正例进行预测
            ans = ans + [1 for i in range(len(res_pos))]  # 添加正例的标签
            score.append(res_pos)

            res_neg = self.test_one_step(neg_ins)  # 对负例进行预测
            ans = ans + [0 for i in range(len(res_pos))]  # 添加负例的标签
            score.append(res_neg)

        # 合并预测得分和标签
        score = np.concatenate(score, axis = -1)
        ans = np.array(ans)

        # 如果没有提供阈值，则计算最佳阈值
        if threshlod == None:
            threshlod, _ = self.get_best_threshlod(score, ans)

        # 计算在指定阈值下的准确率
        res = np.concatenate([ans.reshape(-1,1), score.reshape(-1,1)], axis = -1)
        order = np.argsort(score)
        res = res[order]

        total_all = (float)(len(score))
        total_current = 0.0
        total_true = np.sum(ans)
        total_false = total_all - total_true

        # 遍历排序后的结果，根据阈值计算准确率
        for index, [ans, score] in enumerate(res):
            if score > threshlod:
                acc = (2 * total_current + total_false - index) / total_all
                break
            elif ans == 1:
                total_current += 1.0

        return acc, threshlod  # 返回分类准确率和阈值