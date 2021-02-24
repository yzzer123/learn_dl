"""
用自己想法实现的感知器
用numpy库加速向量运算迭代
"""
import numpy as np

from functools import reduce

def numpy_test():
    
    a = np.array([1,2,3,4])
    b = np.array([2,3,4,5]) 
    # 按元素进行乘法
    print(list(a * b))
    # 按元素进行相加
    print(list(a + b))
    # 按元素进行倍乘
    print(list(2 * a))
    # 求内积
    print(int(a @ b))

"""
感知器的训练过程可以理解为随机梯度下降
"""
class Perception:
    def __init__(self, activator, weight_size):
        self.activator = activator
        self.weights = np.zeros(weight_size)
        self.bias = 0.

    def predict(self, sample_feature):
        sf_np = np.array(sample_feature)
        return self.activator((sf_np @ self.weights) + self.bias)

    def train(self, features, targets, stepSize=0.1, iteration_times=10):
        for i in range(iteration_times):
            self._one_iteration(features,targets,stepSize)
        return self.acc(features,targets)

    def _one_iteration(self,features,targets,stepSize):
        zipper = np.array(list(zip( features,targets)))
        np.random.shuffle(zipper)
        zipper = list(zipper)
        for feature,target in zipper:
            self._adjust(feature,target,stepSize)
        
    def _adjust(self,feature,target,stepSize):
        delta = target - self.predict(feature)
        self.weights = self.weights + stepSize * delta * np.array(feature)
        self.bias += stepSize * delta

    def acc(self,samples,targets):
        """
        获得感知机准确率
        """
        predicts = [self.predict(sample) for sample in samples]
        accuracy = len(list(filter(lambda v:v[0] == v[1], zip(targets,predicts))))
        accuracy /= len(samples)
        return  accuracy


    def __str__(self):
        return "weight:\t %s\n bias:\t %f\n" % (self.weights, self.bias)

# 激活函数
def f(x):
    return 1 if x > 0 else 0

# 获取训练数据
def get_train_dataset():
    # and 数据集
    # samples = [
    #     (1,1),(1,0),(0,1),(0,0)
    # ]
    # targets = [1,0,0,0]

    # or 数据集
    samples = [
        (1,1),(1,0),(0,1),(0,0)
    ]
    targets = [1,1,1,0]
    return samples,targets

# 训练
def train():
    samples, targets = get_train_dataset()
    and_perception = Perception(f,len(samples[0]))
    print("训练精度",and_perception.train(samples,targets,0.07,10))
    test(and_perception, samples)

# 测试
def test(and_perception,samples):
    print(and_perception)
    for sample in samples:
        print("%d  or  %d  =  %d" % (sample[0], sample[1], and_perception.predict(sample)))


if __name__=="__main__":
    # vector_test()
    train()

