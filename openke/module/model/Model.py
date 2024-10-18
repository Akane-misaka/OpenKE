import torch
import torch.nn as nn
from ..BaseModule import BaseModule


# 定义 Model 类，继承自 BaseModule，作为所有模型的基础类
class Model(BaseModule):

	# 初始化函数，定义实体总数和关系总数
	def __init__(self, ent_tot, rel_tot):
		super(Model, self).__init__()  # 调用父类 BaseModule 的初始化方法
		self.ent_tot = ent_tot  # 存储实体的总数量
		self.rel_tot = rel_tot  # 存储关系的总数量

	# 定义前向传播方法（需要子类实现）
	def forward(self):
		raise NotImplementedError  # 抛出未实现错误，要求子类实现该方法

	# 定义预测方法（需要子类实现）
	def predict(self):
		raise NotImplementedError  # 抛出未实现错误，要求子类实现该方法