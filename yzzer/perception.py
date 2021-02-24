"""
    感知机：
        y = f(w*x+b)  
                f为激活函数  w为权重向量 x为样本 b为偏置项
    
    感知机的训练算法：
        感知机的训练策略是，遇到一个样本，调整一次权重，训练几轮后得出结果
        1. 先计算出 y 和 t的差值  t - y  
                t代表样本的实际标签值(label) 该值记为delta
            当delta>0说明y太小了，整体权重要加大
            当delta<0说明y太大了，整体权重就要减小
        2. 对于权重 w而言 w = w + rate * delta * x    
                rate为训练速率
        3. 对于偏置项b b = b + rate * delta 
                偏置项可以认为是 x永远为1的权重
"""
from functools import reduce;

class VectorsOp:

    @staticmethod
    def element_multiply(x,y):
        """
        将x y 两组向量，相应位置的元素相乘
        返回一个新的向量
        首先把x[x1,x2,x3...]和y[y1,y2,y3,...]打包在一起
        变成[(x1,y1),(x2,y2),(x3,y3),...]
        然后利用map函数计算[x1*y1, x2*y2, x3*y3]
        """
        return list(map(lambda XY:XY[0]*XY[1],zip(x,y)))

    @staticmethod
    def element_add(x,y):
        """
        将x y两组向量相加，相应位置元素相加
        首先把x[x1,x2,x3...]和y[y1,y2,y3,...]打包在一起
        变成[(x1,y1),(x2,y2),(x3,y3),...]
        然后利用map函数计算[x1+y1, x2+y2, x3+y3]
        """
        return  list(map(lambda XY:XY[0]+XY[1],zip(x,y)))

    @staticmethod
    def dot(x,y):
        """
        计算两个向量x和y的内积
        首先把x[x1,x2,x3...]和y[y1,y2,y3,...]按元素相乘
        变成[x1*y1, x2*y2, x3*y3]
        然后利用reduce求和
        """
        return reduce(lambda x,y:x+y,VectorsOp.element_multiply(x,y))

    @staticmethod
    def scala_multiply(v,s):
        """
        将向量v中的每个元素和标量s相乘 返回一个新的向量
        """
        return list(map(lambda x:x*s,v))

# 测试向量类的静态方法
def vector_test():
    x = [1, 2, 3]
    y = [2, 4, 5]
    print("element_multiply", VectorsOp.element_multiply(x,y))
    print("element_add",VectorsOp.element_add(x,y))
    print("dot",VectorsOp.dot(x,y))
    print("scala_multiply",VectorsOp.scala_multiply(x,2))


class Perception:

    def __init__(self, activator, size):
        # 初始化激活函数
        self.activator = activator
        # 初始化权重数组
        self.weight = [0.]*size
        # 初始化偏置项
        self.bias = 0.

    def __str__(self):
        """
        打印训练后的感知机的权重数组
        """
        return "weight:\t %s\nbias:\t%f\n" % (self.weight, self.bias)

    def predict(self,sample):
        """
        返回根据当前权重值得到的预测值
        """

        return self.activator(
            VectorsOp.dot(self.weight,sample) + self.bias
        )

    def _adjust(self,sample,target,rate):
        """
        针对一个样本进行调整权重
        """
        predict = self.predict(sample)
        delta = target - predict
        self.weight = VectorsOp.element_add(
            self.weight, VectorsOp.scala_multiply(sample,rate * delta)
        )
        self.bias += (rate * delta)

    def _one_iteration(self,samples,targets,rate):
        """
        对一组样本进行一次遍历调整
        """
        for sample,target in zip(samples,targets):
            self._adjust(sample,target,rate)

    def acc(self,samples,targets):
        """
        获得感知机准确率
        """
        predicts = [self.predict(sample) for sample in samples]
        accuracy = len(list(filter(lambda v:v[0] == v[1], zip(targets,predicts))))
        accuracy /= len(samples)
        return  accuracy

    def train(self,samples,targets,rate,times=10):
        """
        根据训练轮次，对样本进行多次训练 返回准确率
        """
        for i in range(times):
            self._one_iteration(samples,targets,rate)

        return self.acc(samples,targets)

    


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