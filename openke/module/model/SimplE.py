import torch
import torch.nn as nn
from .Model import Model

# 定义 SimplE 类，继承自 Model，用于知识图谱嵌入任务
class SimplE(Model):

    # 初始化函数，定义嵌入维度及模型参数
    def __init__(self, ent_tot, rel_tot, dim = 100):
        super(SimplE, self).__init__(ent_tot, rel_tot)  # 调用父类初始化

        # 设置嵌入向量的维度
        self.dim = dim

        # 定义实体和关系的嵌入层
        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)  # 实体嵌入
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)  # 关系嵌入
        self.rel_inv_embeddings = nn.Embedding(self.rel_tot, self.dim)  # 反关系嵌入

        # 使用 Xavier 均匀初始化为嵌入赋初值
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_inv_embeddings.weight.data)

    # 内部计算函数，用于计算平均得分（关系和反关系）
    def _calc_avg(self, h, t, r, r_inv):
        # 计算正向和反向关系的得分，并返回它们的平均值
        return (torch.sum(h * r * t, -1) + torch.sum(h * r_inv * t, -1))/2

    # 内部计算函数，用于单向计算得分
    def _calc_ingr(self, h, r, t):
        # 计算头实体、关系和尾实体的乘积，并返回求和结果
        return torch.sum(h * r * t, -1)

    # 前向传播函数，计算一批数据的得分
    def forward(self, data):
        # 从输入数据中提取头实体、尾实体和关系的索引
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']

        # 获取嵌入向量
        h = self.ent_embeddings(batch_h)  # 头实体的嵌入
        t = self.ent_embeddings(batch_t)  # 尾实体的嵌入
        r = self.rel_embeddings(batch_r)  # 正向关系的嵌入
        r_inv = self.rel_inv_embeddings(batch_r)  # 反向关系的嵌入

        # 计算平均得分
        score = self._calc_avg(h, t, r, r_inv)
        return score  # 返回得分

    # 正则化函数，防止模型过拟合
    def regularization(self, data):
        # 从数据中提取实体和关系的索引
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']

        # 获取嵌入向量
        h = self.ent_embeddings(batch_h)  # 头实体的嵌入
        t = self.ent_embeddings(batch_t)  # 尾实体的嵌入
        r = self.rel_embeddings(batch_r)  # 正向关系的嵌入
        r_inv = self.rel_inv_embeddings(batch_r)  # 反向关系的嵌入
        # 计算正则化项，返回平方和的平均值
        regul = (torch.mean(h ** 2) +
                 torch.mean(t ** 2) +
                 torch.mean(r ** 2) +
                 torch.mean(r_inv ** 2)) / 4
        return regul  # 返回正则化值

    # 预测函数，计算并返回三元组的预测得分
    def predict(self, data):
        # 从输入数据中提取实体和关系的索引
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']

        # 获取嵌入向量
        h = self.ent_embeddings(batch_h)  # 头实体的嵌入
        t = self.ent_embeddings(batch_t)  # 尾实体的嵌入
        r = self.rel_embeddings(batch_r)  # 正向关系的嵌入
        # 计算得分并取负值
        score = -self._calc_ingr(h, r, t)

        # 返回预测的得分，并转换为 numpy 格式
        return score.cpu().data.numpy()