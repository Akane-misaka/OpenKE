import torch
import torch.nn as nn
from .Model import Model

# 定义 RESCAL 类，继承自 Model，用于知识图谱嵌入任务
class RESCAL(Model):

	# 初始化函数，定义模型的维度及嵌入参数
	def __init__(self, ent_tot, rel_tot, dim = 100):
		super(RESCAL, self).__init__(ent_tot, rel_tot)  # 调用父类初始化

		# 嵌入向量的维度
		self.dim = dim

		# 定义实体的嵌入向量和关系的矩阵嵌入
		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)  # 实体嵌入
		self.rel_matrices = nn.Embedding(self.rel_tot, self.dim * self.dim)  # 关系矩阵嵌入

		# 使用 Xavier 均匀初始化嵌入参数
		nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
		nn.init.xavier_uniform_(self.rel_matrices.weight.data)

	# 内部计算函数，用于计算模型的得分
	def _calc(self, h, t, r):
		# 将尾实体 t 调整为 (batch_size, dim, 1) 的形状
		t = t.view(-1, self.dim, 1)

		# 将关系 r 调整为 (batch_size, dim, dim) 的形状
		r = r.view(-1, self.dim, self.dim)

		# 计算 r 与 t 的矩阵乘法
		tr = torch.matmul(r, t)

		# 将结果调整为 (batch_size, dim) 的形状
		tr = tr.view(-1, self.dim)

		# 计算头实体与矩阵乘法结果的点积，并取负作为得分
		return -torch.sum(h * tr, -1)

	# 前向传播函数，用于计算一批数据的得分
	def forward(self, data):
		# 从数据中提取头实体、尾实体和关系的索引
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']

		# 获取对应的嵌入向量和关系矩阵
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_matrices(batch_r)

		# 调用 _calc 函数计算得分
		score = self._calc(h ,t, r)
		return score  # 返回得分

	# 正则化函数，用于防止模型过拟合
	def regularization(self, data):
		# 从数据中提取批次索引
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']

		# 获取对应的嵌入向量和关系矩阵
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_matrices(batch_r)

		# 计算嵌入向量和关系矩阵的平方和的均值作为正则化项
		regul = (torch.mean(h ** 2) + torch.mean(t ** 2) +
				 torch.mean(r ** 2)) / 3

		return regul  # 返回正则化值

	# 预测函数，用于返回预测的三元组得分
	def predict(self, data):
		# 调用 forward 计算得分，并取负值
		score = -self.forward(data)
		# 将得分转换为 numpy 格式并返回
		return score.cpu().data.numpy()