import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torch


#root_dir = '/projects/vode/team1/C3D'
root_dir = '/home/dahee333/tmp'
segment_list, labels = [], []
for (root, dir, files) in sorted(os.walk(root_dir)):

    if len(files) == 16:
        print("# root : " , root)
        segment_list.append(root)
        #labels.append(root.split("/")[5])
    #     for file_name in  files:
    #         if ".jpg" in file_name:
    #             path_list.append(root+'/'+file_name)
    #             print("# root : " + root)
    #             print("path : " + root+'/'+file_name)
    # else:
    #     continue
with open('list.txt', 'w', encoding='UTF-8') as f:
    for seg in segment_list:
        f.write(seg+'\n')

print(len(segment_list))



 
