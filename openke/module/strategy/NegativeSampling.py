from .Strategy import Strategy


# 定义 NegativeSampling 类，继承自自定义的 Strategy 类，用于负采样训练策略
class NegativeSampling(Strategy):

    # 初始化函数，接受模型、损失函数、批次大小、正则化系数等参数
    def __init__(self, model=None, loss=None, batch_size=256, regul_rate=0.0, l3_regul_rate=0.0):
        super(NegativeSampling, self).__init__()  # 调用父类的初始化方法
        self.model = model  # 训练中使用的模型
        self.loss = loss  # 损失函数，用于计算正负样本之间的差异
        self.batch_size = batch_size  # 每次训练的批次大小
        self.regul_rate = regul_rate  # L2正则化系数，用于控制模型复杂度
        self.l3_regul_rate = l3_regul_rate  # L3正则化系数，适用于某些模型的正则化

    # 获取正样本得分
    def _get_positive_score(self, score):
        # 提取前 batch_size 大小的数据作为正样本得分
        positive_score = score[:self.batch_size]
        # 调整得分的形状，以便后续计算
        positive_score = positive_score.view(-1, self.batch_size).permute(1, 0)
        return positive_score

    # 获取负样本得分
    def _get_negative_score(self, score):
        # 提取 batch_size 之后的数据作为负样本得分
        negative_score = score[self.batch_size:]
        # 调整得分的形状，以便后续计算
        negative_score = negative_score.view(-1, self.batch_size).permute(1, 0)
        return negative_score

    # 前向传播函数，计算损失值
    def forward(self, data):
        # 使用模型对数据进行前向计算，得到正负样本的得分
        score = self.model(data)
        # 获取正样本和负样本的得分
        p_score = self._get_positive_score(score)
        n_score = self._get_negative_score(score)
        # 使用损失函数计算正负样本之间的损失
        loss_res = self.loss(p_score, n_score)
        # 如果正则化系数不为零，则加上正则化项
        if self.regul_rate != 0:
            # L2正则化项，通过模型的 regularization 方法计算
            loss_res += self.regul_rate * self.model.regularization(data)
        # 如果 L3 正则化系数不为零，则加上 L3 正则化项
        if self.l3_regul_rate != 0:
            # L3正则化项，通过模型的 l3_regularization 方法计算
            loss_res += self.l3_regul_rate * self.model.l3_regularization()
        # 返回总的损失值
        return loss_res
