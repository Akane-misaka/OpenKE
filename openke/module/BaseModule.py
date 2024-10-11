import torch
import torch.nn as nn
import os
import json
import numpy as np

# 定义 BaseModule 类，继承自 nn.Module，作为模型的基础模块
class BaseModule(nn.Module):

	# 初始化函数，定义一些常数参数
	def __init__(self):
		super(BaseModule, self).__init__()  # 调用父类的初始化方法
		# 定义一个常数0，用于后续操作，不可学习
		self.zero_const = nn.Parameter(torch.Tensor([0]))
		self.zero_const.requires_grad = False # 设置不可学习

		# 定义π常数，用于后续计算，不可学习
		self.pi_const = nn.Parameter(torch.Tensor([3.14159265358979323846]))
		self.pi_const.requires_grad = False  # 设置不可学习

	# 加载模型的检查点（模型参数），用于恢复训练好的模型
	def load_checkpoint(self, path):
		# 加载指定路径下的模型参数到当前模型
		self.load_state_dict(torch.load(os.path.join(path)))
		# 将模型设置为评估模式，禁用训练时的行为如 dropout
		self.eval()

	# 保存模型的检查点，将当前模型的参数保存到指定路径
	def save_checkpoint(self, path):
		# 保存模型参数到指定路径
		torch.save(self.state_dict(), path)

	# 从指定路径加载模型参数，参数以 JSON 格式存储
	def load_parameters(self, path):
		# 打开并读取 JSON 文件中的参数
		f = open(path, "r")
		parameters = json.loads(f.read())
		f.close()
		# 将每个参数转换为张量（Tensor）
		for i in parameters:
			parameters[i] = torch.Tensor(parameters[i])
		# 加载这些参数到当前模型，不严格匹配（允许缺少部分参数）
		self.load_state_dict(parameters, strict = False)
		# 将模型设置为评估模式
		self.eval()

	# 将当前模型的参数保存到指定路径，以 JSON 格式存储
	def save_parameters(self, path):
		# 打开文件并写入当前模型的参数
		f = open(path, "w")
		# 获取参数，并以 JSON 格式保存
		f.write(json.dumps(self.get_parameters("list")))
		f.close()

	# 获取模型的参数，支持多种格式（numpy、list、tensor）
	def get_parameters(self, mode = "numpy", param_dict = None):
		# 获取当前模型的所有参数字典
		all_param_dict = self.state_dict()
		# 如果未指定参数字典，则获取所有参数的键
		if param_dict == None:
			param_dict = all_param_dict.keys()

		# 创建结果字典，用于存储转换后的参数
		res = {}
		for param in param_dict:
			# 根据 mode 参数选择不同的格式进行转换
			if mode == "numpy":
				# 转换为 numpy 数组格式
				res[param] = all_param_dict[param].cpu().numpy()
			elif mode == "list":
				# 转换为 Python 列表格式
				res[param] = all_param_dict[param].cpu().numpy().tolist()
			else:
				# 保留为张量（Tensor）格式
				res[param] = all_param_dict[param]
		return res

	# 设置模型的参数，将给定的参数字典加载到模型中
	def set_parameters(self, parameters):
		# 将参数字典中的每个参数转换为张量（Tensor）
		for i in parameters:
			parameters[i] = torch.Tensor(parameters[i])
		# 加载这些参数到当前模型，不严格匹配（允许缺少部分参数）
		self.load_state_dict(parameters, strict = False)
		# 将模型设置为评估模式
		self.eval()