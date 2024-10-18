import torch
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model

# 定义 TransD 类，继承自 Model，用于知识图谱嵌入
class TransD(Model):

	# 初始化函数，定义模型的维度及参数
	def __init__(self, ent_tot, rel_tot, dim_e = 100, dim_r = 100, p_norm = 1, norm_flag = True, margin = None, epsilon = None):
		super(TransD, self).__init__(ent_tot, rel_tot)  # 调用父类初始化

		# 初始化嵌入维度和超参数
		self.dim_e = dim_e  # 实体的嵌入维度
		self.dim_r = dim_r  # 关系的嵌入维度
		self.margin = margin  # 边界值
		self.epsilon = epsilon  # 调整嵌入范围的参数
		self.norm_flag = norm_flag  # 是否进行归一化
		self.p_norm = p_norm  # 范数的类型（默认为 L1 范数）

		# 定义嵌入层：实体、关系及其转移向量
		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim_e)
		self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim_r)
		self.ent_transfer = nn.Embedding(self.ent_tot, self.dim_e)  # 实体的转移向量
		self.rel_transfer = nn.Embedding(self.rel_tot, self.dim_r)  # 关系的转移向量

		# 初始化嵌入向量，若未指定 margin 和 epsilon，则使用 Xavier 均匀初始化
		if margin == None or epsilon == None:
			nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
			nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
			nn.init.xavier_uniform_(self.ent_transfer.weight.data)
			nn.init.xavier_uniform_(self.rel_transfer.weight.data)
		else:
			# 根据 margin 和 epsilon 初始化嵌入的范围
			self.ent_embedding_range = nn.Parameter(
				torch.Tensor([(self.margin + self.epsilon) / self.dim_e]), requires_grad=False
			)
			self.rel_embedding_range = nn.Parameter(
				torch.Tensor([(self.margin + self.epsilon) / self.dim_r]), requires_grad=False
			)
			nn.init.uniform_(
				tensor = self.ent_embeddings.weight.data, 
				a = -self.ent_embedding_range.item(), 
				b = self.ent_embedding_range.item()
			)
			nn.init.uniform_(
				tensor = self.rel_embeddings.weight.data, 
				a= -self.rel_embedding_range.item(), 
				b= self.rel_embedding_range.item()
			)
			nn.init.uniform_(
				tensor = self.ent_transfer.weight.data, 
				a= -self.ent_embedding_range.item(), 
				b= self.ent_embedding_range.item()
			)
			nn.init.uniform_(
				tensor = self.rel_transfer.weight.data, 
				a= -self.rel_embedding_range.item(), 
				b= self.rel_embedding_range.item()
			)

		# 设置 margin 参数是否可训练
		if margin != None:
			self.margin = nn.Parameter(torch.Tensor([margin]))
			self.margin.requires_grad = False  # 不需要训练
			self.margin_flag = True  # 边界值标志
		else:
			self.margin_flag = False

	# 用于调整张量形状的函数
	def _resize(self, tensor, axis, size):
		shape = tensor.size()  # 获取张量形状
		osize = shape[axis]  # 原始维度大小

		# 如果大小相同，则返回原张量
		if osize == size:
			return tensor

		# 如果原始大小大于目标大小，则进行裁剪
		if (osize > size):
			return torch.narrow(tensor, axis, 0, size)

		# 否则，进行填充操作
		paddings = []
		for i in range(len(shape)):
			if i == axis:
				paddings = [0, size - osize] + paddings  # 在指定轴进行填充
			else:
				paddings = [0, 0] + paddings  # 其他轴不填充
		print (paddings)
		return F.pad(tensor, paddings = paddings, mode = "constant", value = 0)

	# 计算模型的得分函数
	def _calc(self, h, t, r, mode):
		# 若开启归一化，则进行 L2 归一化
		if self.norm_flag:
			h = F.normalize(h, 2, -1)
			r = F.normalize(r, 2, -1)
			t = F.normalize(t, 2, -1)

		# 根据 mode 调整张量的形状
		if mode != 'normal':
			h = h.view(-1, r.shape[0], h.shape[-1])
			t = t.view(-1, r.shape[0], t.shape[-1])
			r = r.view(-1, r.shape[0], r.shape[-1])

		# 根据模式计算得分
		if mode == 'head_batch':
			score = h + (r - t)  # 计算头实体批次的得分
		else:
			score = (h + r) - t  # 计算尾实体批次的得分

		# 使用指定范数计算得分
		score = torch.norm(score, self.p_norm, -1).flatten()
		return score

	# 计算实体转移后的嵌入向量
	def _transfer(self, e, e_transfer, r_transfer):
		if e.shape[0] != r_transfer.shape[0]:
			e = e.view(-1, r_transfer.shape[0], e.shape[-1])
			e_transfer = e_transfer.view(-1, r_transfer.shape[0], e_transfer.shape[-1])
			r_transfer = r_transfer.view(-1, r_transfer.shape[0], r_transfer.shape[-1])

			# 调整后的嵌入向量，并进行归一化
			e = F.normalize(
				self._resize(e, -1, r_transfer.size()[-1]) + torch.sum(e * e_transfer, -1, True) * r_transfer,
				p = 2, 
				dim = -1
			)			
			return e.view(-1, e.shape[-1])
		else:
			return F.normalize(
				self._resize(e, -1, r_transfer.size()[-1]) + torch.sum(e * e_transfer, -1, True) * r_transfer,
				p = 2, 
				dim = -1
			)

	# 前向传播函数，计算一批数据的得分
	def forward(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		mode = data['mode']

		# 获取嵌入向量
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)

		# 获取转移向量并进行转移计算
		h_transfer = self.ent_transfer(batch_h)
		t_transfer = self.ent_transfer(batch_t)
		r_transfer = self.rel_transfer(batch_r)
		h = self._transfer(h, h_transfer, r_transfer)
		t = self._transfer(t, t_transfer, r_transfer)

		# 计算得分
		score = self._calc(h ,t, r, mode)
		if self.margin_flag:
			return self.margin - score  # 使用边界值计算得分
		else:
			return score

	# 正则化函数，计算正则化项，避免模型过拟合
	def regularization(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		h_transfer = self.ent_transfer(batch_h)
		t_transfer = self.ent_transfer(batch_t)
		r_transfer = self.rel_transfer(batch_r)

		# 计算正则化值（平方和均值）
		regul = (torch.mean(h ** 2) +
				 torch.mean(t ** 2) + 
				 torch.mean(r ** 2) + 
				 torch.mean(h_transfer ** 2) + 
				 torch.mean(t_transfer ** 2) + 
				 torch.mean(r_transfer ** 2)) / 6
		return regul

	# 预测函数，返回预测得分
	def predict(self, data):
		score = self.forward(data)
		if self.margin_flag:
			score = self.margin - score
			return score.cpu().data.numpy()
		else:
			return score.cpu().data.numpy()