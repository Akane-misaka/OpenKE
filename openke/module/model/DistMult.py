import torch
import torch.nn as nn
from .Model import Model

# 定义 DistMult 类，继承自 Model，适用于知识图谱嵌入任务
class DistMult(Model):

	# 初始化函数，定义嵌入维度、超参数和模型参数
	def __init__(self, ent_tot, rel_tot, dim = 100, margin = None, epsilon = None):
		super(DistMult, self).__init__(ent_tot, rel_tot)  # 调用父类初始化

		# 嵌入向量的维度
		self.dim = dim
		self.margin = margin  # 边界值（用于正则化的控制）
		self.epsilon = epsilon  # 用于调整嵌入范围

		# 定义实体和关系的嵌入向量
		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)  # 实体嵌入
		self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)  # 关系嵌入

		# 判断是否提供 margin 和 epsilon 参数，选择不同的初始化方法
		if margin == None or epsilon == None:
			# 使用 Xavier 均匀初始化，为嵌入向量赋初值
			nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
			nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
		else:
			# 计算嵌入范围，并使用均匀分布初始化
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

	# 内部计算函数，用于计算模型的得分
	def _calc(self, h, t, r, mode):
		# 当模式不是 'normal' 时，调整张量的形状
		if mode != 'normal':
			h = h.view(-1, r.shape[0], h.shape[-1])
			t = t.view(-1, r.shape[0], t.shape[-1])
			r = r.view(-1, r.shape[0], r.shape[-1])

		# 根据模式选择不同的计算方式
		if mode == 'head_batch':
			score = h * (r * t)  # 头实体批次计算
		else:
			score = (h * r) * t  # 尾实体批次计算

		# 在最后一个维度上求和，并将结果展开为一维
		score = torch.sum(score, -1).flatten()
		return score

	# 前向传播函数，计算一批数据的得分
	def forward(self, data):
		# 从输入数据中提取头实体、尾实体和关系的索引及模式
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		mode = data['mode']

		# 获取实体和关系的嵌入向量
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)

		# 调用 _calc 函数计算得分
		score = self._calc(h ,t, r, mode)
		return score  # 返回得分

	# 正则化函数，用于避免模型过拟合
	def regularization(self, data):
		# 提取批次中的实体和关系嵌入
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)

		# 计算每个嵌入向量的平方和的均值作为正则化项
		regul = (torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2)) / 3
		return regul  # 返回正则化值

	# L3 正则化函数，用于进一步限制嵌入向量的大小
	def l3_regularization(self):
		# 计算实体和关系嵌入的 L3 范数，并返回总和
		return (self.ent_embeddings.weight.norm(p = 3)**3 +
				self.rel_embeddings.weight.norm(p = 3)**3)

	# 预测函数，用于返回预测的三元组得分
	def predict(self, data):
		# 调用 forward 函数计算得分，并取负值
		score = -self.forward(data)
		# 将得分转换为 numpy 数组，并返回
		return score.cpu().data.numpy()
