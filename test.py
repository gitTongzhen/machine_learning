import torch
# x[batch_size,channels,height_1,width_1]
# conv[channels,output,height,width]
# 输出 [2,8,7-2+1,3-3+1]
x = torch.randn(2,1,7,7)
print(type(x))
conv = torch.nn.Conv2d(1,8,(3,3))
maxpool = torch.nn.MaxPool2d(2,2)
res = conv(x)
res1 = maxpool(res)
print(res1.shape)
print(res.shape)