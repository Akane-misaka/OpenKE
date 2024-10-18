import torch
import torch.nn as nn
from .Model import Model
import numpy
from numpy import fft

# 定义 HolE 类，继承自 Model，用于知识图谱嵌入任务
class HolE(Model):

	# 初始化函数，定义嵌入维度、超参数和模型参数
	def __init__(self, ent_tot, rel_tot, dim = 100, margin = None, epsilon = None):
		super(HolE, self).__init__(ent_tot, rel_tot)  # 调用父类初始化

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

	# 复数共轭函数
	def _conj(self, tensor):
		# 创建用于共轭操作的矩阵
		zero_shape = (list)(tensor.shape)
		one_shape = (list)(tensor.shape)
		zero_shape[-1] = 1
		one_shape[-1] -= 1
		ze = torch.zeros(size = zero_shape, device = tensor.device)  # 全 0 张量
		on = torch.ones(size = one_shape, device = tensor.device)  # 全 1 张量
		matrix = torch.cat([ze, on], -1)  # 将 0 和 1 连接成矩阵
		matrix = 2 * matrix  # 矩阵乘以 2
		return tensor - matrix * tensor  # 返回共轭张量

	# 提取张量的实部
	def _real(self, tensor):
		dimensions = len(tensor.shape)  # 获取张量的维度
		return tensor.narrow(dimensions - 1, 0, 1)  # 返回实部

	# 提取张量的虚部
	def _imag(self, tensor):
		dimensions = len(tensor.shape)  # 获取张量的维度
		return tensor.narrow(dimensions - 1, 1, 1)  # 返回虚部

	# 复数乘法
	def _mul(self, real_1, imag_1, real_2, imag_2):
		real = real_1 * real_2 - imag_1 * imag_2  # 实部计算
		imag = real_1 * imag_2 + imag_1 * real_2  # 虚部计算
		return torch.cat([real, imag], -1)  # 合并为复数张量

	# 使用循环相关计算向量相似性
	def _ccorr(self, a, b):
		# 计算向量的快速傅里叶变换（FFT），并进行复共轭
		a = self._conj(torch.rfft(a, signal_ndim = 1, onesided = False))
		b = torch.rfft(b, signal_ndim = 1, onesided = False)

		# 进行复数乘法
		res = self._mul(self._real(a), self._imag(a), self._real(b), self._imag(b))

		# 计算逆傅里叶变换
		res = torch.ifft(res, signal_ndim = 1)

		# 返回结果的实部
		return self._real(res).flatten(start_dim = -2)

	# 内部计算函数，用于计算模型的得分
	def _calc(self, h, t, r, mode):
		# 根据模式调整张量形状
		if mode != 'normal':
			h = h.view(-1, r.shape[0], h.shape[-1])
			t = t.view(-1, r.shape[0], t.shape[-1])
			r = r.view(-1, r.shape[0], r.shape[-1])

		# 计算循环相关得分
		score = self._ccorr(h, t) * r
		score = torch.sum(score, -1).flatten()
		return score

	def forward(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		mode = data['mode']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		score = self._calc(h ,t, r, mode)
		return score

	# 正则化函数，避免模型过拟合
	def regularization(self, data):
		# 提取实体和关系嵌入
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)

		# 计算嵌入向量的平方和均值
		regul = (torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2)) / 3
		return regul  # 返回正则化值

	# L3 正则化函数，进一步限制嵌入向量的大小
	def l3_regularization(self):
		# 计算实体和关系嵌入的 L3 范数，并返回总和
		return (self.ent_embeddings.weight.norm(p = 3)**3 +
				self.rel_embeddings.weight.norm(p = 3)**3)

	# 预测函数，返回三元组的得分
	def predict(self, data):
		# 计算得分并取负
		score = -self.forward(data)
		# 返回得分的 numpy 数组
		return score.cpu().data.numpy()
