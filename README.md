# PMF
复现概率矩阵分解（PMF）算法

学号：2020223045112
姓名：周煜坤

PMF.py包括三个功能函数load_data,calculate_rmse,train。

load_data函数：读取MovieLens 100k数据集，划分训练集和测试集，传入train函数

calculate_rmse函数：计算均方根误差，传入train函数

train函数：初始化U,V矩阵，进行U,V,loss,rmse的不断更新，迭代，并打印运行过程

运行截图如下：
![image](https://user-images.githubusercontent.com/70565722/114396125-c58fc000-9bcf-11eb-953c-bf506177d48c.png)
