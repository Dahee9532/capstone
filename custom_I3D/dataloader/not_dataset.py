import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision import transforms


class VideoDataset(Dataset):
    def read_data_set(self):

        #class_names = sorted(os.walk(self.root_dir).__next__()[1])
        class_names = sorted(os.listdir(self.root_dir))
        #print(class_names)
        all_segment_list, all_labels = [], []

        for (root, dir, files) in sorted(os.walk(self.root_dir)):

            if 'normal' in root and len(files) == 16:
                #print("# root : " + root)
                #print("# label : ", root.split("/")[5])
                all_labels.append(root.split("/")[4])
                all_segment_list.append(root)
            else:
                continue

        return all_segment_list, all_labels, len(all_segment_list), len(class_names)

    def __init__(self, root_dir='/home/juyoung0927/team1', num_channel = 3, clip_len=16):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.num_channel = num_channel

        self.all_segment_list, self.all_labels, self.seg_length, self.num_classes = self.read_data_set()
        self.resize_height = 112
        self.resize_width = 112 

        self.label_to_index = {label : index for index, label in enumerate(set(self.all_labels))}
        self.label_array = np.array([self.label_to_index[label] for label in self.all_labels], dtype=int)


    def __len__(self):
        return self.seg_length
#tensor, resize

    def __getitem__(self, index):
        buffer = self.load_frames(self.all_segment_list[index])
        # (16, 112, 112, 3)
        labels = np.array(self.label_array[index])
        torch_buffer = self.to_tensor(buffer)
        # (1, 3, 16, 112, 112)

        return torch_buffer, torch.from_numpy(labels), self.all_segment_list[index]

  
    def to_tensor(self, buffer): # 
        buffer_trans  = buffer.transpose((3, 0, 1, 2))
        buffer_trans= torch.from_numpy(buffer_trans)
        #buffer_trans= buffer_trans.unsqueeze(dim=0)

        return buffer_trans

    # 3 0 1 2 --> 3 16 112 112
    def load_frames(self, seg_dir):
        frames = sorted([os.path.join(seg_dir, img) for img in os.listdir(seg_dir)])

        buffer = np.empty((self.clip_len, self.resize_height, self.resize_width, 3), np.dtype('float32'))

        for i, frame_name in enumerate(frames):
            frame = cv2.imread(frame_name)
            frame = cv2.resize(frame, (self.resize_width, self.resize_height))
            frame = np.array(frame).astype(np.float64)
            buffer[i] = frame #result 112 112 3 

        return buffer

if __name__ == "__main__":
    from torch.utils.data import DataLoader
#/projects/vode/team1/C3D
#/projects/vode/team1/tmp
    train_data = VideoDataset(root_dir='/home/juyoung0927/team1', num_channel = 3, clip_len=16)
    train_loader = DataLoader(train_data, batch_size=10, shuffle=True, num_workers=4)
    
    for i, sample in enumerate(train_loader):
        inputs = sample[0]
        labels = sample[1]
        print(inputs.size())
        print(labels)

        if i == 1:
            break

#3 1 16 112 112
    


'''

root_dir = '/projects/vode/team1/C3D'
#root_dir = '/projects/vode/team1/tmp'
class_names = sorted(os.walk(root_dir).__next__()[1])
print(class_names)
# segment_list, labels = [], []
# for (root, dir, files) in sorted(os.walk(root_dir)):
    
#     if 'anomaly' in root:
#         labels.append(root.split("/")[5])
#         segment_list.append(root)
#         # print("# root : " + root)
#         # print("# label : ", root.split("/")[5])

#     else:
#         continue

#     # if len(dir) > 0:
#     #     for dir_name in dir:
#     #         if 'anomaly' in dir_name:
#     #             print("dir : " + dir_name)
#     #         else: 
#     #             continue

#     # if len(files) > 0:
#     #     for file_name in  files:
#     #         if ".jpg" in file_name:
#     #             path_list.append(root+'/'+file_name)
#     #             print("path : " + root+'/'+file_name)
#     #         else:
#     #             continue
# print(len(segment_list))
# print(len(labels))
'''
