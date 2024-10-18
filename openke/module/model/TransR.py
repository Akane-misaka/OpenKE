import torch
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model

# 定义 TransR 类，继承自 Model，用于知识图谱嵌入任务
class TransR(Model):

	# 初始化函数，定义模型的参数和嵌入层
	def __init__(self, ent_tot, rel_tot, dim_e = 100, dim_r = 100, p_norm = 1, norm_flag = True, rand_init = False, margin = None):
		super(TransR, self).__init__(ent_tot, rel_tot)  # 调用父类初始化

		# 初始化模型参数
		self.dim_e = dim_e  # 实体嵌入维度
		self.dim_r = dim_r  # 关系嵌入维度
		self.norm_flag = norm_flag  # 是否进行归一化
		self.p_norm = p_norm  # 范数类型（如 L1 或 L2）
		self.rand_init = rand_init  # 是否随机初始化转移矩阵

		# 定义实体和关系的嵌入层
		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim_e)
		self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim_r)

		# 使用 Xavier 初始化嵌入
		nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
		nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

		# 定义关系的转移矩阵
		self.transfer_matrix = nn.Embedding(self.rel_tot, self.dim_e * self.dim_r)

		# 如果不使用随机初始化，则将转移矩阵初始化为单位矩阵
		if not self.rand_init:
			identity = torch.zeros(self.dim_e, self.dim_r)  # 创建零矩阵
			for i in range(min(self.dim_e, self.dim_r)):
				identity[i][i] = 1  # 设置对角线为 1，生成单位矩阵
			identity = identity.view(self.dim_r * self.dim_e)  # 展平为一维张量
			for i in range(self.rel_tot):
				self.transfer_matrix.weight.data[i] = identity  # 初始化每个关系的转移矩阵
		else:
			# 如果启用随机初始化，则使用 Xavier 初始化转移矩阵
			nn.init.xavier_uniform_(self.transfer_matrix.weight.data)

		# 设置 margin 参数是否启用
		if margin != None:
			self.margin = nn.Parameter(torch.Tensor([margin]))
			self.margin.requires_grad = False  # 设置为不可训练
			self.margin_flag = True  # 启用 margin 标志
		else:
			self.margin_flag = False

	# 计算得分函数，用于计算实体与关系的得分
	def _calc(self, h, t, r, mode):
		# 如果需要归一化，则对实体和关系向量进行 L2 归一化
		if self.norm_flag:
			h = F.normalize(h, 2, -1)
			r = F.normalize(r, 2, -1)
			t = F.normalize(t, 2, -1)

		# 调整张量的形状以适应批量计算
		if mode != 'normal':
			h = h.view(-1, r.shape[0], h.shape[-1])
			t = t.view(-1, r.shape[0], t.shape[-1])
			r = r.view(-1, r.shape[0], r.shape[-1])

		# 根据模式计算得分
		if mode == 'head_batch':
			score = h + (r - t)  # 计算头实体批次的得分
		else:
			score = (h + r) - t  # 计算尾实体批次的得分

		# 使用指定的范数计算得分
		score = torch.norm(score, self.p_norm, -1).flatten()
		return score

	# 转移函数，将实体嵌入映射到关系的空间中
	def _transfer(self, e, r_transfer):
		r_transfer = r_transfer.view(-1, self.dim_e, self.dim_r)  # 调整转移矩阵的形状

		# 根据实体和转移矩阵的形状进行不同方式的矩阵乘法
		if e.shape[0] != r_transfer.shape[0]:
			e = e.view(-1, r_transfer.shape[0], self.dim_e).permute(1, 0, 2)  # 调整实体的形状
			e = torch.matmul(e, r_transfer).permute(1, 0, 2)  # 进行矩阵乘法并调整形状
		else:
			e = e.view(-1, 1, self.dim_e)  # 调整实体的形状
			e = torch.matmul(e, r_transfer)  # 进行矩阵乘法
		return e.view(-1, self.dim_r)  # 返回映射后的实体嵌入

	# 前向传播函数，计算一批数据的得分
	def forward(self, data):
		batch_h = data['batch_h']  # 获取头实体索引
		batch_t = data['batch_t']  # 获取尾实体索引
		batch_r = data['batch_r']  # 获取关系索引
		mode = data['mode']  # 模式（如 normal, head_batch）

		# 获取对应的嵌入向量和转移矩阵
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		r_transfer = self.transfer_matrix(batch_r)

		# 将实体嵌入映射到关系空间
		h = self._transfer(h, r_transfer)
		t = self._transfer(t, r_transfer)

		# 计算得分
		score = self._calc(h ,t, r, mode)
		if self.margin_flag:
			return self.margin - score  # 使用 margin 调整得分
		else:
			return score

	# 正则化函数，计算正则化项，防止模型过拟合
	def regularization(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']

		# 获取嵌入向量和转移矩阵
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		r_transfer = self.transfer_matrix(batch_r)

		# 计算正则化项（平方和的平均值）
		regul = (torch.mean(h ** 2) +
				 torch.mean(t ** 2) + 
				 torch.mean(r ** 2) +
				 torch.mean(r_transfer ** 2)) / 4
		return regul * regul  # 返回正则化值

	# 预测函数，返回预测的得分
	def predict(self, data):
		score = self.forward(data)  # 计算得分
		if self.margin_flag:
			score = self.margin - score  # 使用 margin 调整得分
			return score.cpu().data.numpy()  # 返回 numpy 格式的得分
		else:
			return score.cpu().data.numpy()