import torch
import torch.autograd as autograd
import torch.nn as nn
from .Model import Model

# 定义 RotatE 类，继承自 Model，用于知识图谱嵌入任务
class RotatE(Model):

	# 初始化函数，定义模型的维度、超参数和嵌入参数
	def __init__(self, ent_tot, rel_tot, dim = 100, margin = 6.0, epsilon = 2.0):
		super(RotatE, self).__init__(ent_tot, rel_tot)  # 调用父类初始化

		# 初始化模型超参数
		self.margin = margin  # 边界值
		self.epsilon = epsilon  # 调整嵌入范围的参数

		# 定义实体和关系的嵌入维度
		self.dim_e = dim * 2  # 实体嵌入的维度（旋转表示需要双倍维度）
		self.dim_r = dim  # 关系嵌入的维度

		# 定义实体和关系的嵌入层
		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim_e)  # 实体嵌入层
		self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim_r)  # 关系嵌入层

		# 定义实体嵌入的范围，并初始化参数
		self.ent_embedding_range = nn.Parameter(
			torch.Tensor([(self.margin + self.epsilon) / self.dim_e]), 
			requires_grad=False
		)

		# 使用均匀分布初始化实体嵌入
		nn.init.uniform_(
			tensor = self.ent_embeddings.weight.data, 
			a=-self.ent_embedding_range.item(), 
			b=self.ent_embedding_range.item()
		)

		# 定义关系嵌入的范围，并初始化参数
		self.rel_embedding_range = nn.Parameter(
			torch.Tensor([(self.margin + self.epsilon) / self.dim_r]), 
			requires_grad=False
		)

		# 使用均匀分布初始化关系嵌入
		nn.init.uniform_(
			tensor = self.rel_embeddings.weight.data, 
			a=-self.rel_embedding_range.item(), 
			b=self.rel_embedding_range.item()
		)

		# 定义不可训练的 margin 参数
		self.margin = nn.Parameter(torch.Tensor([margin]))
		self.margin.requires_grad = False  # 设置为不可训练

	# 内部计算函数，用于计算模型的得分
	def _calc(self, h, t, r, mode):
		pi = self.pi_const  # 圆周率 π

		# 将实体嵌入拆分为实部和虚部
		re_head, im_head = torch.chunk(h, 2, dim=-1)  # 头实体
		re_tail, im_tail = torch.chunk(t, 2, dim=-1)  # 尾实体

		# 计算关系的相位向量
		phase_relation = r / (self.rel_embedding_range.item() / pi)

		re_relation = torch.cos(phase_relation)  # 实部
		im_relation = torch.sin(phase_relation)  # 虚部

		# 调整嵌入的形状以适应批处理计算
		re_head = re_head.view(-1, re_relation.shape[0], re_head.shape[-1]).permute(1, 0, 2)
		re_tail = re_tail.view(-1, re_relation.shape[0], re_tail.shape[-1]).permute(1, 0, 2)
		im_head = im_head.view(-1, re_relation.shape[0], im_head.shape[-1]).permute(1, 0, 2)
		im_tail = im_tail.view(-1, re_relation.shape[0], im_tail.shape[-1]).permute(1, 0, 2)
		im_relation = im_relation.view(-1, re_relation.shape[0], im_relation.shape[-1]).permute(1, 0, 2)
		re_relation = re_relation.view(-1, re_relation.shape[0], re_relation.shape[-1]).permute(1, 0, 2)

		# 根据模式计算得分
		if mode == "head_batch":
			# 如果是 head_batch 模式，计算头实体的旋转后得分
			re_score = re_relation * re_tail + im_relation * im_tail
			im_score = re_relation * im_tail - im_relation * re_tail
			re_score = re_score - re_head
			im_score = im_score - im_head
		else:
			# 如果是 tail_batch 模式，计算尾实体的旋转后得分
			re_score = re_head * re_relation - im_head * im_relation
			im_score = re_head * im_relation + im_head * re_relation
			re_score = re_score - re_tail
			im_score = im_score - im_tail

		# 将实部和虚部的得分堆叠并计算范数
		score = torch.stack([re_score, im_score], dim = 0)
		score = score.norm(dim = 0).sum(dim = -1)
		return score.permute(1, 0).flatten()  # 返回计算的得分

	# 前向传播函数，计算一批数据的得分
	def forward(self, data):
		# 从数据中提取头实体、尾实体和关系的索引及模式
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		mode = data['mode']

		# 获取实体和关系的嵌入
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)

		# 计算得分并减去 margin
		score = self.margin - self._calc(h ,t, r, mode)
		return score  # 返回得分

	# 预测函数，用于返回预测的三元组得分
	def predict(self, data):
		score = -self.forward(data)  # 调用 forward 计算得分并取负值
		return score.cpu().data.numpy()  # 返回 numpy 格式的得分

	# 正则化函数，用于防止模型过拟合
	def regularization(self, data):
		# 从数据中提取批次的实体和关系索引
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']

		# 获取对应的嵌入
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)

		# 计算嵌入的平方和的平均值作为正则化项
		regul = (torch.mean(h ** 2) +
				 torch.mean(t ** 2) + 
				 torch.mean(r ** 2)) / 3
		return regul  # 返回正则化值