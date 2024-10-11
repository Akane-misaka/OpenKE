import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from .Loss import Loss

# 定义 MarginLoss 类，继承自自定义的 Loss 类，用于计算损失
class MarginLoss(Loss):

	# 初始化函数，接受可选的对抗温度（adv_temperature）和默认的边界值（margin）
	def __init__(self, adv_temperature = None, margin = 6.0):
		super(MarginLoss, self).__init__()  # 调用父类的初始化方法
		# 定义边界值参数，并设置不可学习
		self.margin = nn.Parameter(torch.Tensor([margin]))
		self.margin.requires_grad = False

		# 如果提供了对抗温度参数，则设置对抗相关属性
		if adv_temperature != None:
			self.adv_temperature = nn.Parameter(torch.Tensor([adv_temperature]))
			self.adv_temperature.requires_grad = False
			self.adv_flag = True  # 设置标志，表示使用对抗训练
		else:
			self.adv_flag = False  # 否则，不使用对抗训练
	
	def get_weights(self, n_score):
		# 根据负样本得分计算权重，使用对抗采样策略
		return F.softmax(-n_score * self.adv_temperature, dim = -1).detach()

	# 前向传播函数，计算损失值
	def forward(self, p_score, n_score):
		# 如果启用了对抗训练，则使用加权的方式计算损失
		if self.adv_flag:
			# 对负样本得分加权后计算 p_score 和 n_score 之间的损失
			return (self.get_weights(n_score) * torch.max(p_score - n_score, -self.margin)).sum(dim = -1).mean() + self.margin
		else:
			# 普通模式下的损失计算，不使用加权
			return (torch.max(p_score - n_score, -self.margin)).mean() + self.margin


	# 预测函数，用于返回损失的数值
	def predict(self, p_score, n_score):
		# 计算损失并返回numpy格式的结果
		score = self.forward(p_score, n_score)
		return score.cpu().data.numpy()