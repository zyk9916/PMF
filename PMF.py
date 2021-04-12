import numpy as np

class PMF():

    def load_data(self, path, train_ratio):   # 读取数据，生成评分矩阵
        user_set = {}                   # 用户集合
        item_set = {}                   # 商品集合
        user_index = 0                  # 用户索引
        item_index = 0                  # 商品索引
        data = []                       # 评分矩阵
        with open(path) as f:
            for line in f.readlines():
                u, i, r, _ = line.split()                           # u:用户序号，i:商品序号，r:评分，_:冗余数据
                if u not in user_set:
                    user_set[u] = user_index
                    user_index += 1
                if i not in item_set:
                    item_set[i] = item_index
                    item_index += 1
                data.append([user_set[u], item_set[i], float(r)])   # data每一行结构为[用户索引，商品索引，评分]

        np.random.shuffle(data)                                     # 打乱data中行的顺序，行内不变
        trainset = data[0:int(len(data) * train_ratio)]             # 训练集分片
        testset = data[int(len(data) * train_ratio):]               # 测试集分片
        return user_index, item_index, trainset, testset            # 数据读取完毕后，user_index,item_index分别表示用户总数和商品总数

    def train(self,num_user,num_item,trainset,testset,learning_rate,K,regu_u,regu_i,max_iteration):
        U = np.random.normal(0,0.1,(num_user,K))        # 生成一个num_user * K的矩阵，元素为服从均值为0，方差为0.1的高斯分布的随机数
        V = np.random.normal(0,0.1,(num_item,K))        # 生成一个num_item * K的矩阵，元素为服从均值为0，方差为0.1的高斯分布的随机数
        max_rmse = 100.0
        endure_count = 3
        patience = 0
        for iter in range(max_iteration):
            loss = 0.0                                  # 损失初始化
            for data in trainset:                          # 每个data是一个三维向量，分别赋值给user,item,rating
                user = data[0]
                item = data[1]
                rating = data[2]
                predict_rating = np.dot(U[user],V[item].T)                                      # .T操作取转置，numpy.dot计算点积
                error = rating - predict_rating                                                 # 计算偏差
                U[user] += learning_rate * (error * V[item] - regu_u * U[user])                 # 更新U，V
                V[item] += learning_rate * (error * U[user] - regu_i * V[item])
                loss += error ** 2
                loss += regu_u * np.square(U[user]).sum() + regu_i * np.square(V[item]).sum()   # 更新loss
            loss = 0.5 * loss
            rmse = self.calculate_rmse(U,V,testset)                                                # 计算均方根误差
            print('iteration:%d   loss:%.3f   rmse:%.5f'%(iter,loss,rmse))
            if rmse < max_rmse:                                                                 # 提前停止
                max_rmse = rmse
                patience = 0
            else:
                patience += 1
            if patience >= endure_count:                      # 连续多次均方根误差不再下降时，结束迭代
                break

    def calculate_rmse(self,U,V,testset):
        test_count = len(testset)
        tmp_rmse = 0.0
        for t in testset:
            user = t[0]
            item = t[1]
            rating = t[2]
            predict_rating = np.dot(U[user],V[item].T)
            tmp_rmse += np.square(rating - predict_rating)
        rmse = np.sqrt(tmp_rmse/test_count)                   # 均方根误差
        return rmse


if __name__=='__main__':
    pmf = PMF()
    num_user,num_item,trainset,testset = pmf.load_data('ml-100k/u.data',0.8)    # 读取数据，训练集:测试集=8:2
    pmf.train(num_user,num_item,trainset,testset,0.01,10,0.01,0.01,100)         # 学习率设为0.1，潜在特征个数设为10，U,V更新参数设为0.01，最大迭代次数设为100