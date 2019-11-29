import os
import shutil

path = "F:\\BaiduNetdiskDownload\\kaggle\\train\\train"
# goal_path = "D:\\Desktop\\workspace\\github\\machine_learning\\machine_learning\\VGGDataSet\\val\\cat"
goal_path = "D:\\Desktop\\workspace\\github\\machine_learning\\machine_learning\\VGGDataSet\\val\\dog"
current_fold = os.listdir()
## 拷贝cat文件
# for i in range(2000,2500):
#     image_file_path = os.path.join(path ,"cat.{}.jpg".format(i))
#     print(image_file_path)
#     image_goal_path = os.path.join(goal_path,"cat.{}.jpg".format(i))
#     shutil.copy(image_file_path,image_goal_path)

for i in range(2000,2500):
    image_file_path = os.path.join(path ,"dog.{}.jpg".format(i))
    print(image_file_path)
    image_goal_path = os.path.join(goal_path,"dog.{}.jpg".format(i))
    shutil.copy(image_file_path,image_goal_path)