# coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import sys
import datetime
import ctypes
import json
import numpy as np
import copy
from tqdm import tqdm

# 定义 Trainer 类，用于训练模型
class Trainer(object):

	# 初始化函数，接收模型、数据加载器、训练次数、学习率、是否使用GPU等参数
	def __init__(self,
				 model = None,
				 data_loader = None,
				 train_times = 1000,
				 alpha = 0.5,  # 学习率
				 use_gpu = True,  # 是否使用GPU
				 opt_method = "sgd",  # 优化方法，默认使用SGD
				 save_steps = None,  # 保存模型的步长
				 checkpoint_dir = None):  # 模型保存路径

		self.work_threads = 8  # 线程数
		self.train_times = train_times  # 训练次数

		self.opt_method = opt_method  # 优化器方法
		self.optimizer = None  # 初始化优化器为空
		self.lr_decay = 0  # 学习率衰减
		self.weight_decay = 0  # 权重衰减
		self.alpha = alpha  # 学习率

		self.model = model  # 模型
		self.data_loader = data_loader  # 数据加载器
		self.use_gpu = use_gpu  # 是否使用GPU
		self.save_steps = save_steps  # 保存模型的频率
		self.checkpoint_dir = checkpoint_dir  # 模型检查点保存的目录

	# 单步训练函数，进行一次梯度更新
	def train_one_step(self, data):
		self.optimizer.zero_grad()  # 清空梯度
		# 前向传播计算损失
		loss = self.model({
			'batch_h': self.to_var(data['batch_h'], self.use_gpu),  # 转换头实体数据为Variable
			'batch_t': self.to_var(data['batch_t'], self.use_gpu),  # 转换尾实体数据为Variable
			'batch_r': self.to_var(data['batch_r'], self.use_gpu),  # 转换关系数据为Variable
			'batch_y': self.to_var(data['batch_y'], self.use_gpu),  # 转换标签数据为Variable
			'mode': data['mode']  # 模式（如训练模式）
		})
		loss.backward()  # 反向传播计算梯度
		self.optimizer.step()  # 更新模型参数
		return loss.item()  # 返回损失值

	# 主训练流程
	def run(self):
		# 如果使用GPU，则将模型移动到GPU
		if self.use_gpu:
			self.model.cuda()

		# 初始化优化器，如果未定义，则根据选择的优化方法设置
		if self.optimizer != None:
			pass
		elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
			self.optimizer = optim.Adagrad(
				self.model.parameters(),
				lr=self.alpha,
				lr_decay=self.lr_decay,
				weight_decay=self.weight_decay,
			)
		elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
			self.optimizer = optim.Adadelta(
				self.model.parameters(),
				lr=self.alpha,
				weight_decay=self.weight_decay,
			)
		elif self.opt_method == "Adam" or self.opt_method == "adam":
			self.optimizer = optim.Adam(
				self.model.parameters(),
				lr=self.alpha,
				weight_decay=self.weight_decay,
			)
		else:
			# 默认使用SGD优化器
			self.optimizer = optim.SGD(
				self.model.parameters(),
				lr = self.alpha,
				weight_decay=self.weight_decay,
			)
		print("Finish initializing...")  # 打印初始化完成信息

		# 使用tqdm显示训练进度条，遍历训练次数
		training_range = tqdm(range(self.train_times))
		for epoch in training_range:
			res = 0.0  # 累积当前轮次的总损失
			for data in self.data_loader:
				loss = self.train_one_step(data)  # 训练一步并获取损失
				res += loss  # 累加损失
			# 在进度条中显示当前轮次和平均损失
			training_range.set_description("Epoch %d | loss: %f" % (epoch, res))

			# 每隔指定的步数保存一次模型检查点
			if self.save_steps and self.checkpoint_dir and (epoch + 1) % self.save_steps == 0:
				print("Epoch %d has finished, saving..." % (epoch))
				self.model.save_checkpoint(os.path.join(self.checkpoint_dir + "-" + str(epoch) + ".ckpt"))

	# 设置新的模型
	def set_model(self, model):
		self.model = model

	# 将numpy数组转换为PyTorch的Variable，并根据是否使用GPU移动到相应设备
	def to_var(self, x, use_gpu):
		if use_gpu:
			return Variable(torch.from_numpy(x).cuda())
		else:
			return Variable(torch.from_numpy(x))

	# 设置是否使用GPU
	def set_use_gpu(self, use_gpu):
		self.use_gpu = use_gpu

	# 设置学习率
	def set_alpha(self, alpha):
		self.alpha = alpha

	# 设置学习率衰减
	def set_lr_decay(self, lr_decay):
		self.lr_decay = lr_decay

	# 设置权重衰减
	def set_weight_decay(self, weight_decay):
		self.weight_decay = weight_decay

	# 设置优化方法
	def set_opt_method(self, opt_method):
		self.opt_method = opt_method

	# 设置训练次数
	def set_train_times(self, train_times):
		self.train_times = train_times

	# 设置模型保存步长和保存路径
	def set_save_steps(self, save_steps, checkpoint_dir = None):
		self.save_steps = save_steps
		if not self.checkpoint_dir:
			self.set_checkpoint_dir(checkpoint_dir)

	# 设置模型保存的目录
	def set_checkpoint_dir(self, checkpoint_dir):
		self.checkpoint_dir = checkpoint_dir