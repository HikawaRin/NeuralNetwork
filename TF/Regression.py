import tensorflow as TF
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

DataSet = pd.read_csv("Data/housing.csv", sep=' ')
data = np.array(DataSet.values, dtype=np.float)

# 归一化
dmin = []
dmax = []
for i in range(data.shape[1]):
    dmin.append(data[:, i].min())
    dmax.append(data[:, i].max())
    data[:, i] = (data[:, i] - dmin[i]) / (dmax[i] - dmin[i])

X = data[:, 0:(data.shape[1]-1)]
Y = data[:, (data.shape[1]-1)]

x = TF.placeholder(TF.float32, [None, 13], name="X")
y = TF.placeholder(TF.float32, [None, 1], name="Y")

with TF.name_scope("Model"):
    w = TF.Variable(TF.random_normal([13, 1], stddev=0.01), name="w")
    b = TF.Variable(1.0, name="b")

    def model(x, w, b):
        return TF.matmul(x, w) + b

    pred = model(x, w, b)

train_epochs = 50
learning_rate = 0.001

with TF.name_scope("LossFunction"):
    loss_function = TF.reduce_mean(TF.pow(y - pred, 2))

optimizer = TF.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

sess = TF.Session()
init = TF.global_variables_initializer()
sess.run(init)

loss_list = []
for epoch in range(train_epochs):
    loss_sum = 0.0
    for xs,ys in zip(X,Y):
        
        xs = xs.reshape(1,13)
        ys = ys.reshape(1,1)
        
        _,loss = sess.run([optimizer,loss_function],feed_dict={x:xs,y:ys})
        
        loss_sum = loss_sum + loss
    
        # loss_list.append(loss)     #每步添加一次
    
    X,Y = shuffle(X,Y)
    
    b0temp = b.eval(session=sess)            #训练中当前变量b值
    w0temp = w.eval(session=sess)            #训练中当前权重w值
    loss_average = loss_sum/len(Y)      #当前训练中的平均损失
    
    loss_list.append(loss_average)              
    print("epoch=",epoch+1,"loss=",loss_average,"b=",b0temp,"w=",w0temp)

plt.plot(loss_list)
plt.show()

n = np.random.randint(506)       
print(n)
x_test = X[n]

x_test = x_test.reshape(1,13)
predict = sess.run(pred,feed_dict={x:x_test})
print("预测值：%f"%predict)

target = Y[n]
print("标签值：%f"%target)