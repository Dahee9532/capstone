
import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision import transforms

#/projects/vode/team1/16_croki_segment/fight/10-4_cam01_fight03_place07_night_spring_resized.mp4/10-4_cam01_fight03_place07_night_spring_resized.mp4_0348_normal
#/projects/vode/team1/16_segment/fight/1-1_cam01_fight04_place02_night_summer_resized.mp4/1-1_cam01_fight04_place02_night_summer_resized.mp4_0218_normal


class VideoDataset(Dataset):
    def read_data_set(self, txt_path, chrom_txt_path):

        f = open(txt_path, 'r')
        lines = f.readlines() #리스트로 만들어줌
        lines = [line.rstrip('\n') for line in lines]

        all_segment_list, all_labels = [], []

        for i in sorted(lines):
            if(len(os.listdir(i)) < 16):
               continue 
            if i.endswith("normal"):
                all_labels.append("normal")
            elif i.endswith("anomaly"):
                all_labels.append(i.split("/")[5])

            all_segment_list.append(i)

        f.close()

        f = open(chrom_txt_path, 'r')
        lines = f.readlines() #리스트로 만들어줌
        lines = [line.rstrip('\n') for line in lines]

        for i in sorted(lines):
            if(len(os.listdir(i)) < 16):
               continue 
            if i.endswith("normal"):
                all_labels.append("normal")
            elif i.endswith("anomaly"):
                all_labels.append(i.split("/")[5])

            all_segment_list.append(i)

        f.close()

        class_names = []
        for v in all_labels:
            if v not in class_names:
                class_names.append(v)
        #print(sorted(class_names))

        return all_segment_list, all_labels, len(all_segment_list), len(class_names)

    def __init__(self, txt_path = '/home/dahee333/custom_X3D/list_txt/train/root_train_list.txt',chrom_txt_path = '/home/dahee333/custom_X3D/list_txt/train/root_chrom_train_list.txt',  num_channel = 3, clip_len=16):
        self.clip_len = clip_len
        self.num_channel = num_channel

        self.all_segment_list, self.all_labels, self.seg_length, self.num_classes = self.read_data_set(txt_path, chrom_txt_path)
        self.resize_height = 224
        self.resize_width = 224

        self.label_to_index = {label : index for index, label in enumerate(set(self.all_labels))}
        self.label_array = np.array([self.label_to_index[label] for label in self.all_labels], dtype=int)


    def __len__(self):
        return self.seg_length

    def __getitem__(self, index):
        video_segment_path = self.all_segment_list[index]
        buffer = self.load_frames(video_segment_path)

        # (16, 112, 112, 3)
        labels = np.array(self.label_array[index])
        torch_buffer = self.to_tensor(buffer)
        # (1, 3, 16, 112, 112)

        return torch_buffer, torch.from_numpy(labels), video_segment_path

  
    def to_tensor(self, buffer): 
        buffer_trans  = buffer.transpose((3, 0, 1, 2))
        buffer_trans= torch.from_numpy(buffer_trans)

        return buffer_trans

    # 3 0 1 2 --> 3 16 112 112
    def load_frames(self, seg_dir):
        frames = sorted([os.path.join(seg_dir, img) for img in os.listdir(seg_dir)])

        buffer = np.empty((self.clip_len, self.resize_height, self.resize_width, 3), np.dtype('float32'))

        for i, frame_name in enumerate(frames):
            frame = cv2.imread(frame_name)
            frame = (frame - frame.min())/(frame.max() - frame.min())
            frame = cv2.resize(frame, (self.resize_width, self.resize_height))
            frame = np.array(frame).astype(np.float64)
            buffer[i] = frame #result 112 112 3 

        return buffer

if __name__ == "__main__":
    from torch.utils.data import DataLoader
#/projects/vode/team1/C3D
#/projects/vode/team1/tmp
    train_data = VideoDataset(txt_path = '/home/dahee333/two_custom_X3D/list_txt/val/16_val_list.txt', chrom_txt_path = '/home/dahee333/two_custom_X3D/list_txt/val/16_val_chro_list.txt', num_channel = 3, clip_len=16)
    train_loader = DataLoader(train_data, batch_size=10, shuffle=True, num_workers=4)
    
    for i, sample in enumerate(train_loader):
        inputs = sample[0]
        labels = sample[1]
        seg_path = sample[2]

        print(labels)
        print(seg_path, "\n")






