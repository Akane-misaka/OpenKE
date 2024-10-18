import torch
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model

# 定义 TransH 类，继承自 Model，用于知识图谱嵌入任务
class TransH(Model):

	# 初始化函数，定义模型的参数和嵌入层
	def __init__(self, ent_tot, rel_tot, dim = 100, p_norm = 1, norm_flag = True, margin = None, epsilon = None):
		super(TransH, self).__init__(ent_tot, rel_tot)  # 调用父类初始化

		# 初始化模型的参数和超参数
		self.dim = dim  # 嵌入维度
		self.margin = margin  # 边界值，用于损失计算
		self.epsilon = epsilon  # 嵌入调整范围
		self.norm_flag = norm_flag  # 是否进行归一化
		self.p_norm = p_norm  # 范数类型（如 L1 或 L2）

		# 定义嵌入层：实体、关系和关系法向量
		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
		self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)
		self.norm_vector = nn.Embedding(self.rel_tot, self.dim)  # 用于关系超平面的法向量

		# 初始化嵌入参数
		if margin == None or epsilon == None:
			# 使用 Xavier 均匀分布初始化嵌入
			nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
			nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
			nn.init.xavier_uniform_(self.norm_vector.weight.data)
		else:
			# 根据 margin 和 epsilon 初始化嵌入的范围
			self.embedding_range = nn.Parameter(
				torch.Tensor([(self.margin + self.epsilon) / self.dim]), requires_grad=False
			)
			nn.init.uniform_(
				tensor = self.ent_embeddings.weight.data, 
				a = -self.embedding_range.item(), 
				b = self.embedding_range.item()
			)
			nn.init.uniform_(
				tensor = self.rel_embeddings.weight.data, 
				a= -self.embedding_range.item(), 
				b= self.embedding_range.item()
			)
			nn.init.uniform_(
				tensor = self.norm_vector.weight.data, 
				a= -self.embedding_range.item(), 
				b= self.embedding_range.item()
			)

		# 设置 margin 参数是否可训练
		if margin != None:
			self.margin = nn.Parameter(torch.Tensor([margin]))
			self.margin.requires_grad = False  # 设置为不可训练
			self.margin_flag = True  # 启用 margin 标志
		else:
			self.margin_flag = False

	# 计算模型得分的函数
	def _calc(self, h, t, r, mode):
		# 如果需要归一化，则进行 L2 归一化
		if self.norm_flag:
			h = F.normalize(h, 2, -1)
			r = F.normalize(r, 2, -1)
			t = F.normalize(t, 2, -1)
		# 根据模式调整张量形状
		if mode != 'normal':
			h = h.view(-1, r.shape[0], h.shape[-1])
			t = t.view(-1, r.shape[0], t.shape[-1])
			r = r.view(-1, r.shape[0], r.shape[-1])

		# 根据不同模式计算得分
		if mode == 'head_batch':
			score = h + (r - t)  # 计算头实体批次的得分
		else:
			score = (h + r) - t  # 计算尾实体批次的得分
		score = torch.norm(score, self.p_norm, -1).flatten()
		return score

	# 计算实体的超平面转移
	def _transfer(self, e, norm):
		# 对关系法向量进行归一化
		norm = F.normalize(norm, p = 2, dim = -1)

		# 调整实体和法向量的形状并计算转移结果
		if e.shape[0] != norm.shape[0]:
			e = e.view(-1, norm.shape[0], e.shape[-1])
			norm = norm.view(-1, norm.shape[0], norm.shape[-1])
			e = e - torch.sum(e * norm, -1, True) * norm
			return e.view(-1, e.shape[-1])
		else:
			return e - torch.sum(e * norm, -1, True) * norm

	# 前向传播函数，计算一批数据的得分
	def forward(self, data):
		batch_h = data['batch_h']  # 头实体索引
		batch_t = data['batch_t']  # 尾实体索引
		batch_r = data['batch_r']  # 关系索引
		mode = data['mode']  # 模式（如 normal, head_batch）

		# 获取嵌入向量和法向量
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		r_norm = self.norm_vector(batch_r)

		# 对实体进行超平面转移
		h = self._transfer(h, r_norm)
		t = self._transfer(t, r_norm)

		# 计算得分
		score = self._calc(h ,t, r, mode)
		if self.margin_flag:
			return self.margin - score  # 使用 margin 计算得分
		else:
			return score

	# 正则化函数，计算正则化项，防止模型过拟合
	def regularization(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']

		# 获取嵌入向量和法向量
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		r_norm = self.norm_vector(batch_r)

		# 计算正则化项（平方和的平均值）
		regul = (torch.mean(h ** 2) +
				 torch.mean(t ** 2) + 
				 torch.mean(r ** 2) +
				 torch.mean(r_norm ** 2)) / 4
		return regul

	# 预测函数，返回预测的得分
	def predict(self, data):
		score = self.forward(data)  # 计算得分
		if self.margin_flag:
			score = self.margin - score
			return score.cpu().data.numpy()  # 使用 margin 调整得分
		else:
			return score.cpu().data.numpy()  # 返回 numpy 格式的结果