import torch
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model

# 定义 TransE 类，继承自 Model，用于知识图谱嵌入任务
class TransE(Model):

	# 初始化函数，定义模型的参数
	def __init__(self, ent_tot, rel_tot, dim = 100, p_norm = 1, norm_flag = True, margin = None, epsilon = None):
		super(TransE, self).__init__(ent_tot, rel_tot)  # 调用父类初始化

		# 初始化模型的参数和超参数
		self.dim = dim  # 嵌入的维度
		self.margin = margin  # 边界值（用于控制模型的学习能力）
		self.epsilon = epsilon  # 控制嵌入向量的范围
		self.norm_flag = norm_flag  # 是否进行归一化
		self.p_norm = p_norm  # 范数类型（L1或L2）

		# 定义实体和关系的嵌入层
		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
		self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)

		# 初始化嵌入参数
		if margin == None or epsilon == None:
			# 使用 Xavier 均匀分布初始化嵌入
			nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
			nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
		else:
			# 根据 margin 和 epsilon 初始化嵌入范围
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

		# 设置 margin 参数是否可训练
		if margin != None:
			self.margin = nn.Parameter(torch.Tensor([margin]))
			self.margin.requires_grad = False  # 设置为不可训练
			self.margin_flag = True  # 启用 margin 标志
		else:
			self.margin_flag = False

	# 计算得分函数，用于计算头实体、关系和尾实体之间的得分
	def _calc(self, h, t, r, mode):
		# 如果需要归一化，则进行 L2 归一化
		if self.norm_flag:
			h = F.normalize(h, 2, -1)
			r = F.normalize(r, 2, -1)
			t = F.normalize(t, 2, -1)

		# 根据 mode 参数调整张量形状
		if mode != 'normal':
			h = h.view(-1, r.shape[0], h.shape[-1])
			t = t.view(-1, r.shape[0], t.shape[-1])
			r = r.view(-1, r.shape[0], r.shape[-1])

		# 根据不同模式计算得分
		if mode == 'head_batch':
			score = h + (r - t)  # 计算头实体批次的得分
		else:
			score = (h + r) - t  # 计算尾实体批次的得分

		# 使用指定的范数计算得分
		score = torch.norm(score, self.p_norm, -1).flatten()
		return score

	# 前向传播函数，计算输入数据的得分
	def forward(self, data):
		batch_h = data['batch_h']  # 头实体索引
		batch_t = data['batch_t']  # 尾实体索引
		batch_r = data['batch_r']  # 关系索引
		mode = data['mode']  # 模式（normal, head_batch, tail_batch）

		# 获取对应的嵌入向量
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)

		# 计算得分
		score = self._calc(h ,t, r, mode)

		# 根据是否启用 margin 返回不同的得分
		if self.margin_flag:
			return self.margin - score
		else:
			return score

	# 正则化函数，用于计算嵌入的正则化项，防止模型过拟合
	def regularization(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']

		# 获取嵌入向量
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)

		# 计算嵌入向量的平方和均值作为正则化项
		regul = (torch.mean(h ** 2) +
				 torch.mean(t ** 2) + 
				 torch.mean(r ** 2)) / 3
		return regul  # 返回正则化值

	# 预测函数，用于计算预测得分
	def predict(self, data):
		score = self.forward(data)  # 调用 forward 函数计算得分
		if self.margin_flag:
			score = self.margin - score  # 使用 margin 调整得分
			return score.cpu().data.numpy()  # 返回 numpy 格式的得分
		else:
			return score.cpu().data.numpy()