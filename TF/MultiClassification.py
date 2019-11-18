import tensorflow as TF
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

DataSet = pd.read_csv("Data/iris.data", header=None)
data = np.array(DataSet.values)

y_label = {"Iris-setosa" : 0, "Iris-versicolor" : 1, "Iris-virginica" : 2}

X = DataSet.values[:, 0:4]
X = np.array(X, dtype=np.float)
Y = DataSet.values[:, 4]
# Y = np.array(list(map(lambda y, y_label=y_label:y_label[y], Y)))
binaryY = []
for y in Y:
    if (y == "Iris-setosa"):
        binaryY.append([1, 0, 0])
    elif (y == "Iris-versicolor"):
        binaryY.append([0, 1, 0])
    else:
        binaryY.append([0, 0, 1])
binaryY = np.array(binaryY)

# 归一化
dmin = []
dmax = []
for i in range(X.shape[1]):
    dmin.append(X[:, i].min())
    dmax.append(X[:, i].max())
    X[:, i] = (X[:, i] - dmin[i]) / (dmax[i] - dmin[i])

x = TF.placeholder(TF.float32, [None, 4], name="X")
y = TF.placeholder(TF.float32, [None, 3], name="Y")

with TF.name_scope("Model"):
    w = TF.Variable(TF.random_normal([4, 3], stddev=0.01), name="w")
    b = TF.Variable(0.0, name="b")

    def model(x, w, b):
        return TF.nn.softmax(TF.matmul(x, w) + b)

    pred = model(x, w, b)

train_epochs = 500
learning_rate = 0.05

with TF.name_scope("LossFunction"):
    loss_function = -TF.reduce_sum(y * TF.log(pred))

optimizer = TF.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

sess = TF.Session()
init = TF.global_variables_initializer()
sess.run(init)

loss_list = []
for epoch in range(train_epochs):
    loss_sum = 0.0
    for xs,ys in zip(X,binaryY):
        
        xs = xs.reshape(1,4)
        ys = ys.reshape(1,3)
        
        _,loss = sess.run([optimizer,loss_function],feed_dict={x:xs,y:ys})
        
        loss_sum = loss_sum + loss
    
        # loss_list.append(loss)     #每步添加一次
    
    X,binaryY = shuffle(X,binaryY)
    
    b0temp = b.eval(session=sess)            #训练中当前变量b值
    w0temp = w.eval(session=sess)            #训练中当前权重w值
    loss_average = loss_sum      #当前训练中的平均损失
    
    loss_list.append(loss_average)              
    print("epoch=",epoch+1,"loss=",loss_average,"b=",b0temp,"w=",w0temp)

plt.plot(loss_list)
plt.show()

n = np.random.randint(150)       
print(n)
x_test = X[n]

x_test = x_test.reshape(1,4)
predict = sess.run(pred,feed_dict={x:x_test})
predict = predict.argmax()
print("预测值：%f"%predict)

target = y_label[Y[n]]
print("标签值：%f"%target)