import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .Loss import Loss

# 定义 SoftplusLoss 类，继承自自定义的 Loss 类，用于计算损失
class SoftplusLoss(Loss):

	# 初始化函数，接受可选的对抗温度（adv_temperature）
	def __init__(self, adv_temperature = None):
		super(SoftplusLoss, self).__init__()  # 调用父类的初始化方法
		# 使用 Softplus 作为损失计算的基础
		self.criterion = nn.Softplus()

		# 如果提供了对抗温度参数，则设置对抗相关属性
		if adv_temperature != None:
			self.adv_temperature = nn.Parameter(torch.Tensor([adv_temperature]))
			self.adv_temperature.requires_grad = False
			self.adv_flag = True  # 将对抗温度作为不可学习的参数
		else:
			self.adv_flag = False  # 未启用对抗训练

	# 根据负样本得分计算权重，使用对抗采样策略
	def get_weights(self, n_score):
		# 对负样本得分进行 softmax 操作，作为权重
		return F.softmax(n_score * self.adv_temperature, dim = -1).detach()

	# 前向传播函数，计算损失值
	def forward(self, p_score, n_score):
		# 如果启用了对抗训练，则使用加权的方式计算损失
		if self.adv_flag:
			# 计算正样本的 Softplus 损失，对负样本的损失根据权重加权求和
			return (self.criterion(-p_score).mean() + (self.get_weights(n_score) * self.criterion(n_score)).sum(dim = -1).mean()) / 2
		else:
			# 普通模式下的损失计算，不使用加权
			return (self.criterion(-p_score).mean() + self.criterion(n_score).mean()) / 2


	# 预测函数，用于返回损失的数值
	def predict(self, p_score, n_score):
		# 计算损失并返回 numpy 格式的结果
		score = self.forward(p_score, n_score)
		return score.cpu().data.numpy()