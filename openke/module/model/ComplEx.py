import torch
import torch.nn as nn
from .Model import Model

# 定义 ComplEx 类，继承自 Model，适用于知识图谱嵌入任务
class ComplEx(Model):
    # 初始化函数，定义嵌入维度和模型参数
    def __init__(self, ent_tot, rel_tot, dim = 100):
        super(ComplEx, self).__init__(ent_tot, rel_tot)  # 调用父类初始化

        # 嵌入向量的维度
        self.dim = dim

        # 定义实体和关系的实部和虚部嵌入
        self.ent_re_embeddings = nn.Embedding(self.ent_tot, self.dim)  # 实体的实部嵌入
        self.ent_im_embeddings = nn.Embedding(self.ent_tot, self.dim)  # 实体的虚部嵌入
        self.rel_re_embeddings = nn.Embedding(self.rel_tot, self.dim)  # 关系的实部嵌入
        self.rel_im_embeddings = nn.Embedding(self.rel_tot, self.dim)  # 关系的虚部嵌入

        # 使用 Xavier 均匀初始化，为嵌入向量赋初值
        nn.init.xavier_uniform_(self.ent_re_embeddings.weight.data)
        nn.init.xavier_uniform_(self.ent_im_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_re_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_im_embeddings.weight.data)

    # 内部计算函数，用于计算模型的得分
    def _calc(self, h_re, h_im, t_re, t_im, r_re, r_im):
        # 根据实体和关系的实部、虚部计算得分
        return torch.sum(
            h_re * t_re * r_re  # 实部与实部相乘
            + h_im * t_im * r_re  # 虚部与实部相乘
            + h_re * t_im * r_im  # 实部与虚部相乘
            - h_im * t_re * r_im,   # 虚部与实部相乘，并相减
            -1  # 在最后一个维度上求和
        )

    # 前向传播函数，用于计算一批数据的得分
    def forward(self, data):
        # 从输入数据中获取头实体、尾实体和关系的索引
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']

        # 根据索引获取对应的嵌入向量
        h_re = self.ent_re_embeddings(batch_h)  # 头实体的实部嵌入
        h_im = self.ent_im_embeddings(batch_h)  # 头实体的虚部嵌入
        t_re = self.ent_re_embeddings(batch_t)  # 尾实体的实部嵌入
        t_im = self.ent_im_embeddings(batch_t)  # 尾实体的虚部嵌入
        r_re = self.rel_re_embeddings(batch_r)  # 关系的实部嵌入
        r_im = self.rel_im_embeddings(batch_r)  # 关系的虚部嵌入

        # 计算三元组的得分
        score = self._calc(h_re, h_im, t_re, t_im, r_re, r_im)
        return score  # 返回得分

    # 正则化函数，避免模型过拟合
    def regularization(self, data):
        # 从数据中提取实体和关系的嵌入
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h_re = self.ent_re_embeddings(batch_h)
        h_im = self.ent_im_embeddings(batch_h)
        t_re = self.ent_re_embeddings(batch_t)
        t_im = self.ent_im_embeddings(batch_t)
        r_re = self.rel_re_embeddings(batch_r)
        r_im = self.rel_im_embeddings(batch_r)

        # 计算嵌入向量的平方和的平均值作为正则化项
        regul = (torch.mean(h_re ** 2) +
                 torch.mean(h_im ** 2) + 
                 torch.mean(t_re ** 2) +
                 torch.mean(t_im ** 2) +
                 torch.mean(r_re ** 2) +
                 torch.mean(r_im ** 2)) / 6  # 归一化处理
        return regul  # 返回正则化值

    # 预测函数，返回三元组的得分，用于模型评估
    def predict(self, data):
        # 调用 forward 函数计算得分，并取反
        score = -self.forward(data)
        # 将得分转换为 numpy 格式返回
        return score.cpu().data.numpy()