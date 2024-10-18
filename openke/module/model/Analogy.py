import torch
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model

# 定义 Analogy 类，继承自 Model，用于知识图谱嵌入任务
class Analogy(Model):

	# 初始化函数，设置实体和关系的嵌入向量
	def __init__(self, ent_tot, rel_tot, dim = 100):
		super(Analogy, self).__init__(ent_tot, rel_tot)  # 调用父类的初始化方法

		# 嵌入向量的维度
		self.dim = dim
		# 定义实体和关系的实部和虚部嵌入
		self.ent_re_embeddings = nn.Embedding(self.ent_tot, self.dim)  # 实体实部嵌入
		self.ent_im_embeddings = nn.Embedding(self.ent_tot, self.dim)  # 实体虚部嵌入
		self.rel_re_embeddings = nn.Embedding(self.rel_tot, self.dim)  # 关系实部嵌入
		self.rel_im_embeddings = nn.Embedding(self.rel_tot, self.dim)  # 关系虚部嵌入

		# 额外的实体和关系嵌入（双倍维度）
		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim * 2)
		self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim * 2)

		# 使用 Xavier 均匀初始化，为嵌入向量赋初值
		nn.init.xavier_uniform_(self.ent_re_embeddings.weight.data)
		nn.init.xavier_uniform_(self.ent_im_embeddings.weight.data)
		nn.init.xavier_uniform_(self.rel_re_embeddings.weight.data)
		nn.init.xavier_uniform_(self.rel_im_embeddings.weight.data)
		nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
		nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

	# 内部计算函数，用于计算模型的得分（score）
	def _calc(self, h_re, h_im, h, t_re, t_im, t, r_re, r_im, r):
		return (-torch.sum(r_re * h_re * t_re +
						   r_re * h_im * t_im +
						   r_im * h_re * t_im -
						   r_im * h_im * t_re, -1)
				-torch.sum(h * t * r, -1))

	# 前向传播函数，计算一批数据的得分
	def forward(self, data):
		# 从输入数据中获取头实体、尾实体和关系的索引
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']

		# 根据索引获取对应的嵌入向量
		h_re = self.ent_re_embeddings(batch_h)  # 头实体的实部嵌入
		h_im = self.ent_im_embeddings(batch_h)  # 头实体的虚部嵌入
		h = self.ent_embeddings(batch_h)  # 头实体的完整嵌入

		t_re = self.ent_re_embeddings(batch_t)  # 尾实体的实部嵌入
		t_im = self.ent_im_embeddings(batch_t)  # 尾实体的虚部嵌入
		t = self.ent_embeddings(batch_t)  # 尾实体的完整嵌入

		r_re = self.rel_re_embeddings(batch_r)  # 关系的实部嵌入
		r_im = self.rel_im_embeddings(batch_r)  # 关系的虚部嵌入
		r = self.rel_embeddings(batch_r)  # 关系的完整嵌入

		# 调用 _calc 函数计算得分
		score = self._calc(h_re, h_im, h, t_re, t_im, t, r_re, r_im, r)
		return score  # 返回得分

	# 正则化函数，用于避免模型过拟合
	def regularization(self, data):
		# 从数据中提取实体和关系的嵌入
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		h_re = self.ent_re_embeddings(batch_h)
		h_im = self.ent_im_embeddings(batch_h)
		h = self.ent_embeddings(batch_h)
		t_re = self.ent_re_embeddings(batch_t)
		t_im = self.ent_im_embeddings(batch_t)
		t = self.ent_embeddings(batch_t)
		r_re = self.rel_re_embeddings(batch_r)
		r_im = self.rel_im_embeddings(batch_r)
		r = self.rel_embeddings(batch_r)

		# 计算每个嵌入的平方和的均值，作为正则化项
		regul = (torch.mean(h_re ** 2) +
				 torch.mean(h_im ** 2) + 
				 torch.mean(h ** 2) + 
				 torch.mean(t_re ** 2) + 
				 torch.mean(t_im ** 2) + 
				 torch.mean(t ** 2) + 
				 torch.mean(r_re ** 2) + 
				 torch.mean(r_im ** 2) + 
				 torch.mean(r ** 2)) / 9  # 将总和归一化处理
		return regul  # 返回正则化值

	# 预测函数，用于返回预测的得分
	def predict(self, data):
		# 调用 forward 函数计算得分
		score = -self.forward(data)
		# 将得分转换为 numpy 数组，并返回
		return score.cpu().data.numpy()