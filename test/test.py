import torch 
import torch.nn as nn

# m = nn.ReLU(inplace = False)
m = nn.Dropout(p = 0.1)
input = torch.randn(7)

print("输出之前的数值",input)

output = m(input)

print("输出之后的数值",output)