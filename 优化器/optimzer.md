pytorch优化器
1.构建优化器
要构建一个Optimizer,你必须给它一个包含参数(必须都是Variable对象)进行优化，然后，您可以
optimizer的参数选项，比如学习率，权重衰减。具体参考torch.optim中文文档
```
optimizer = optim.SGD(model.parameters(),lr = 0.01,momentum = 0.9)
optimizer = optim.Adam([var1,var2],lr = 0.001)

```
差别

```
SGD是最基础的优化方法，普通的训练方法
需要重复不断的把整套数据放入神经网络NN中训练，这样消耗的计算资源很大，当我们使用SGD会把数据拆分后再分批
不断放入NN中计算
```
Momentum传统参数的更新是把原始的w，累加上一个负的学习率(learing rate)乘以校正值(dx)
此方法比较曲折

```

```
AdaGrad 优化学习率，使得每一个参数更新都会有自己与众不同的学习率。与momentum类似，不过
不是给喝醉酒的人安排一个下坡
