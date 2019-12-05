
# tensor 操作常见知识
import torch
# x[batch_size,channels,height_1,width_1]
# conv[channels,output,height,width]
# 输出 [2,8,7-2+1,3-3+1]
x = torch.randn(2,1,3,3)
x_noise = x.new(x.size()).zero_().long()
print(x)
print(x_noise.shape)
print(x.device)
print(x.dtype)
print(x_noise[0][0][0][0])
# conv = torch.nn.Conv2d(1,8,(3,3))
# maxpool = torch.nn.MaxPool2d(2,2)
# res = conv(x)
# res1 = maxpool(res)
# print(res1.shape)
# print(res.shape)

## cat 表示把矩阵拼起来
## 后面的参数为0 的时候是按行拼起来，后面的参数为1的时候表示按列拼起来


import torch

A = torch.ones(2,3)
b = torch.ones(4,3)
C = torch.cat((A,b),0)
print(C.size())


# stack 增加新的维度进行堆叠

a = torch.randn(1,2)
b = torch.randn(1,2)
c = torch.stack((a,b),0)
print(a,b,c)

## permute 适用于多维数据，更加灵活的transpose

x = torch.randn(2,3,4)
print(x.size())
x_shape = x.permute(1,0,2)
print(x_shape.size())

## squeeze和unsqueeze
# squeeze 是去掉元素数量为1的dim_n维度，同理unsqueeze(dim_n),增加dim_n维度
import torch 
b = torch.Tensor(2,1)
b0 =b.unsqueeze(2)
b_ = b.squeeze()
print(b0.size())

## reshape

A = torch.randn(2,3,2)
B = A.reshape(12,1)
print(B)


#torch.argmax(dim) 返回指定维度最大值的序号,dim是指定维度，dim = 0就是返回指定列的，为0就是指定行的
## 同理对于torch.argmin(dim)也是适用的
a = torch.Tensor([[1,5,5,2],
                  [9,-6,2,8],
                  [-3,7,-9,1]])
b = torch.argmax(a,dim=0)
print(b)
print(b.shape)

# tensor常见的取值操作:记住口诀，一行二列

a = torch.Tensor([[1,5,5,2],
                  [9,-6,2,8],
                  [-3,7,-9,1]])
b = a[:-1,:-2]
print(b)

# clamp 函数输出限制在最大值和最小值之间

a = torch.Tensor([[1,5,5,2],
                  [9,-6,2,8],
                  [-3,7,-9,1]])
b = torch.clamp(a,3,7)
print(b)