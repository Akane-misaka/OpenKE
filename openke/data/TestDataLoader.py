# coding:utf-8
import os
import ctypes
import numpy as np

# 定义一个测试数据采样器类，用于生成迭代器
class TestDataSampler(object):

	def __init__(self, data_total, data_sampler):
		# 初始化采样器，保存数据总数和采样函数
		self.data_total = data_total
		self.data_sampler = data_sampler
		self.total = 0  # 用于记录当前迭代次数

	# 实现 __iter__ 方法，使对象可以被迭代
	def __iter__(self):
		return self

	# 实现 __next__ 方法，用于获取下一个采样的数据
	def __next__(self):
		self.total += 1
		# 如果当前次数超过数据总数，则停止迭代
		if self.total > self.data_total:
			raise StopIteration()
		# 返回当前采样器生成的数据
		return self.data_sampler()

	# 实现 __len__ 方法，返回数据总数
	def __len__(self):
		return self.data_total

# 定义一个测试数据加载器类，用于加载测试数据并与 C++ 库进行交互
class TestDataLoader(object):

	def __init__(self, in_path = "./", sampling_mode = 'link', type_constrain = True):
		# 获取C++动态库Base.so的路径，并加载该库
		base_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../release/Base.so"))
		self.lib = ctypes.cdll.LoadLibrary(base_file)

		# 设置 C++ 函数的参数类型（用于链接预测）
		self.lib.getHeadBatch.argtypes = [
			ctypes.c_void_p,
			ctypes.c_void_p,
			ctypes.c_void_p,
		]
		self.lib.getTailBatch.argtypes = [
			ctypes.c_void_p,
			ctypes.c_void_p,
			ctypes.c_void_p,
		]
		# 设置 C++ 函数的参数类型（用于三元组分类）
		self.lib.getTestBatch.argtypes = [
			ctypes.c_void_p,
			ctypes.c_void_p,
			ctypes.c_void_p,
			ctypes.c_void_p,
			ctypes.c_void_p,
			ctypes.c_void_p,
		]
		# 设置测试数据的路径、采样模式和类型约束标志
		self.in_path = in_path
		self.sampling_mode = sampling_mode
		self.type_constrain = type_constrain
		self.read()

	def read(self):
		# 读取并初始化测试数据
		self.lib.setInPath(ctypes.create_string_buffer(self.in_path.encode(), len(self.in_path) * 2))
		self.lib.randReset()  # 重置随机数生成器
		self.lib.importTestFiles()  # 导入测试数据文件

		# 如果启用了类型约束，则导入类型文件
		if self.type_constrain:
			self.lib.importTypeFiles()

		# 获取实体总数、关系总数和测试三元组总数
		self.relTotal = self.lib.getRelationTotal()
		self.entTotal = self.lib.getEntityTotal()
		self.testTotal = self.lib.getTestTotal()

		# 初始化测试时使用的头、尾实体和关系数组
		self.test_h = np.zeros(self.entTotal, dtype=np.int64)
		self.test_t = np.zeros(self.entTotal, dtype=np.int64)
		self.test_r = np.zeros(self.entTotal, dtype=np.int64)
		self.test_h_addr = self.test_h.__array_interface__["data"][0]
		self.test_t_addr = self.test_t.__array_interface__["data"][0]
		self.test_r_addr = self.test_r.__array_interface__["data"][0]

		# 初始化测试正负样本的头、尾实体和关系数组
		self.test_pos_h = np.zeros(self.testTotal, dtype=np.int64)
		self.test_pos_t = np.zeros(self.testTotal, dtype=np.int64)
		self.test_pos_r = np.zeros(self.testTotal, dtype=np.int64)
		self.test_pos_h_addr = self.test_pos_h.__array_interface__["data"][0]
		self.test_pos_t_addr = self.test_pos_t.__array_interface__["data"][0]
		self.test_pos_r_addr = self.test_pos_r.__array_interface__["data"][0]
		self.test_neg_h = np.zeros(self.testTotal, dtype=np.int64)
		self.test_neg_t = np.zeros(self.testTotal, dtype=np.int64)
		self.test_neg_r = np.zeros(self.testTotal, dtype=np.int64)
		self.test_neg_h_addr = self.test_neg_h.__array_interface__["data"][0]
		self.test_neg_t_addr = self.test_neg_t.__array_interface__["data"][0]
		self.test_neg_r_addr = self.test_neg_r.__array_interface__["data"][0]

	# 用于链接预测的采样函数
	def sampling_lp(self):
		res = []
		# 获取头实体批次数据
		self.lib.getHeadBatch(self.test_h_addr, self.test_t_addr, self.test_r_addr)
		res.append({
			"batch_h": self.test_h.copy(), 
			"batch_t": self.test_t[:1].copy(), 
			"batch_r": self.test_r[:1].copy(),
			"mode": "head_batch"
		})
		# 获取尾实体批次数据
		self.lib.getTailBatch(self.test_h_addr, self.test_t_addr, self.test_r_addr)
		res.append({
			"batch_h": self.test_h[:1],
			"batch_t": self.test_t,
			"batch_r": self.test_r[:1],
			"mode": "tail_batch"
		})
		return res

	# 用于三元组分类的采样函数
	def sampling_tc(self):
		# 获取测试批次的正负样本
		self.lib.getTestBatch(
			self.test_pos_h_addr,
			self.test_pos_t_addr,
			self.test_pos_r_addr,
			self.test_neg_h_addr,
			self.test_neg_t_addr,
			self.test_neg_r_addr,
		)
		# 返回包含正负样本的批次数据
		return [ 
			{
				'batch_h': self.test_pos_h,
				'batch_t': self.test_pos_t,
				'batch_r': self.test_pos_r ,
				"mode": "normal"
			}, 
			{
				'batch_h': self.test_neg_h,
				'batch_t': self.test_neg_t,
				'batch_r': self.test_neg_r,
				"mode": "normal"
			}
		]

	"""接口函数用于获取主要参数"""

	def get_ent_tot(self):
		return self.entTotal

	def get_rel_tot(self):
		return self.relTotal

	def get_triple_tot(self):
		return self.testTotal

	# 设置采样模式
	def set_sampling_mode(self, sampling_mode):
		self.sampling_mode = sampling_mode

	# 返回测试数据总数
	def __len__(self):
		return self.testTotal

	# 实现迭代器方法，根据采样模式返回数据采样器
	def __iter__(self):
		if self.sampling_mode == "link":
			self.lib.initTest()  # 初始化测试环境
			return TestDataSampler(self.testTotal, self.sampling_lp)  # 链接预测的采样器
		else:
			self.lib.initTest()
			return TestDataSampler(1, self.sampling_tc)  # 三元组分类的采样器