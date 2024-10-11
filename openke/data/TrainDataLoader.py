# coding:utf-8
import os
import ctypes
import numpy as np

# 定义训练数据采样器类，用于批量生成训练数据
class TrainDataSampler(object):

	def __init__(self, nbatches, datasampler):
		# 初始化采样器，保存批次数量和采样函数
		self.nbatches = nbatches  # 总批次数
		self.datasampler = datasampler  # 采样函数
		self.batch = 0  # 当前批次索引

	# 实现 __iter__ 方法，使对象可以被迭代
	def __iter__(self):
		return self

	# 实现 __next__ 方法，用于获取下一个批次的数据
	def __next__(self):
		self.batch += 1
		# 如果当前批次超过批次数量，则停止迭代
		if self.batch > self.nbatches:
			raise StopIteration()
		# 返回当前批次的数据
		return self.datasampler()

	# 实现 __len__ 方法，返回批次数量
	def __len__(self):
		return self.nbatches

# 定义训练数据加载器类，用于加载和处理训练数据
class TrainDataLoader(object):

	def __init__(self, 
		in_path = "./",
		tri_file = None,
		ent_file = None,
		rel_file = None,
		batch_size = None,
		nbatches = None,
		threads = 8,
		sampling_mode = "normal",
		bern_flag = False,
		filter_flag = True,
		neg_ent = 1,
		neg_rel = 0):

		# 获取C++动态库Base.so的路径，并加载该库
		base_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../release/Base.so"))
		self.lib = ctypes.cdll.LoadLibrary(base_file)

		# 设置C++库中采样函数的参数类型
		self.lib.sampling.argtypes = [
			ctypes.c_void_p,  # 头实体数组地址
			ctypes.c_void_p,  # 尾实体数组地址
			ctypes.c_void_p,  # 关系数组地址
			ctypes.c_void_p,  # 标签数组地址
			ctypes.c_int64,  # 批次大小
			ctypes.c_int64,  # 负采样实体数量
			ctypes.c_int64,  # 负采样关系数量
			ctypes.c_int64,  # 模式（头实体、尾实体替换或普通）
			ctypes.c_int64,  # 是否进行过滤
			ctypes.c_int64,  # 额外参数1
			ctypes.c_int64  # 额外参数2
		]
		# 设置输入路径及数据文件路径
		self.in_path = in_path
		self.tri_file = tri_file
		self.ent_file = ent_file
		self.rel_file = rel_file
		if in_path != None:
			self.tri_file = in_path + "train2id.txt"
			self.ent_file = in_path + "entity2id.txt"
			self.rel_file = in_path + "relation2id.txt"
		# 初始化其他训练参数
		self.work_threads = threads  # 工作线程数
		self.nbatches = nbatches  # 批次数量
		self.batch_size = batch_size  # 每批次大小
		self.bern = bern_flag  # 是否使用Bernoulli负采样
		self.filter = filter_flag  # 是否进行过滤
		self.negative_ent = neg_ent  # 负采样实体数量
		self.negative_rel = neg_rel  # 负采样关系数量
		self.sampling_mode = sampling_mode  # 采样模式（normal、头替换、尾替换）
		self.cross_sampling_flag = 0  # 用于交叉采样
		self.read()  # 读取并初始化训练数据

	# 读取训练数据，并进行初始化
	def read(self):
		# 设置输入路径到C++库中
		if self.in_path != None:
			self.lib.setInPath(ctypes.create_string_buffer(self.in_path.encode(), len(self.in_path) * 2))
		else:
			# 如果没有指定in_path，手动设置各数据文件路径
			self.lib.setTrainPath(ctypes.create_string_buffer(self.tri_file.encode(), len(self.tri_file) * 2))
			self.lib.setEntPath(ctypes.create_string_buffer(self.ent_file.encode(), len(self.ent_file) * 2))
			self.lib.setRelPath(ctypes.create_string_buffer(self.rel_file.encode(), len(self.rel_file) * 2))

		# 设置Bernoulli负采样和工作线程数
		self.lib.setBern(self.bern)
		self.lib.setWorkThreads(self.work_threads)
		self.lib.randReset()  # 重置随机数种子
		self.lib.importTrainFiles()  # 导入训练数据文件
		# 获取关系总数、实体总数和三元组总数
		self.relTotal = self.lib.getRelationTotal()
		self.entTotal = self.lib.getEntityTotal()
		self.tripleTotal = self.lib.getTrainTotal()

		# 根据三元组总数和批次数量计算每批次的大小
		if self.batch_size == None:
			self.batch_size = self.tripleTotal // self.nbatches
		if self.nbatches == None:
			self.nbatches = self.tripleTotal // self.batch_size
		# 设置每个批次的总大小，包括负采样
		self.batch_seq_size = self.batch_size * (1 + self.negative_ent + self.negative_rel)

		# 初始化头实体、尾实体、关系和标签的数组，用于批次采样
		self.batch_h = np.zeros(self.batch_seq_size, dtype=np.int64)
		self.batch_t = np.zeros(self.batch_seq_size, dtype=np.int64)
		self.batch_r = np.zeros(self.batch_seq_size, dtype=np.int64)
		self.batch_y = np.zeros(self.batch_seq_size, dtype=np.float32)
		# 获取这些数组的内存地址，传递给C++库使用
		self.batch_h_addr = self.batch_h.__array_interface__["data"][0]
		self.batch_t_addr = self.batch_t.__array_interface__["data"][0]
		self.batch_r_addr = self.batch_r.__array_interface__["data"][0]
		self.batch_y_addr = self.batch_y.__array_interface__["data"][0]

	# 普通模式下的负采样函数
	def sampling(self):
		# 调用C++库的采样函数，生成负采样的批次数据
		self.lib.sampling(
			self.batch_h_addr,
			self.batch_t_addr,
			self.batch_r_addr,
			self.batch_y_addr,
			self.batch_size,
			self.negative_ent,
			self.negative_rel,
			0,  # 模式：0表示普通模式
			self.filter,
			0,
			0
		)
		# 返回当前批次的头实体、尾实体、关系和标签数据
		return {
			"batch_h": self.batch_h, 
			"batch_t": self.batch_t, 
			"batch_r": self.batch_r, 
			"batch_y": self.batch_y,
			"mode": "normal"
		}

	# 生成头实体替换的负采样
	def sampling_head(self):
		self.lib.sampling(
			self.batch_h_addr,
			self.batch_t_addr,
			self.batch_r_addr,
			self.batch_y_addr,
			self.batch_size,
			self.negative_ent,
			self.negative_rel,
			-1,  # 模式：-1表示头实体替换
			self.filter,
			0,
			0
		)
		return {
			"batch_h": self.batch_h,
			"batch_t": self.batch_t[:self.batch_size],
			"batch_r": self.batch_r[:self.batch_size],
			"batch_y": self.batch_y,
			"mode": "head_batch"
		}

	# 生成尾实体替换的负采样
	def sampling_tail(self):
		self.lib.sampling(
			self.batch_h_addr,
			self.batch_t_addr,
			self.batch_r_addr,
			self.batch_y_addr,
			self.batch_size,
			self.negative_ent,
			self.negative_rel,
			1,  # 模式：1表示尾实体替换
			self.filter,
			0,
			0
		)
		return {
			"batch_h": self.batch_h[:self.batch_size],
			"batch_t": self.batch_t,
			"batch_r": self.batch_r[:self.batch_size],
			"batch_y": self.batch_y,
			"mode": "tail_batch"
		}

	# 交替进行头实体和尾实体替换的负采样
	def cross_sampling(self):
		self.cross_sampling_flag = 1 - self.cross_sampling_flag 
		if self.cross_sampling_flag == 0:
			return self.sampling_head()
		else:
			return self.sampling_tail()

	"""接口函数用于设置主要参数"""

	def set_work_threads(self, work_threads):
		self.work_threads = work_threads

	def set_in_path(self, in_path):
		self.in_path = in_path

	def set_nbatches(self, nbatches):
		self.nbatches = nbatches

	def set_batch_size(self, batch_size):
		self.batch_size = batch_size
		self.nbatches = self.tripleTotal // self.batch_size

	def set_ent_neg_rate(self, rate):
		self.negative_ent = rate

	def set_rel_neg_rate(self, rate):
		self.negative_rel = rate

	def set_bern_flag(self, bern):
		self.bern = bern

	def set_filter_flag(self, filter):
		self.filter = filter

	"""接口函数获取基本参数"""

	def get_batch_size(self):
		return self.batch_size

	def get_ent_tot(self):
		return self.entTotal

	def get_rel_tot(self):
		return self.relTotal

	def get_triple_tot(self):
		return self.tripleTotal

	def __iter__(self):
		if self.sampling_mode == "normal":
			return TrainDataSampler(self.nbatches, self.sampling)
		else:
			return TrainDataSampler(self.nbatches, self.cross_sampling)

	def __len__(self):
		return self.nbatches