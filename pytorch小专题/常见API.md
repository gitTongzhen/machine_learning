
## pytorch 常见API
1.ModuleList是将子Module作为一个List来保存的，可以通过下标进行访问，类似于Python中的List，但是该类需要手动实现forward函数。
self.vgg = nn.ModuleList(self.vgg)
# nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
