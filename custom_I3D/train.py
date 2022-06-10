import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch.nn as nn
import torchfile
from utils import convert_param,activity_name,pool_feature
from torch.utils.data import DataLoader
from dataloader.dataset import VideoDataset
from network import I3D as i3d
from network import C3D as c3d
from network import P3D as p3d
from matplotlib import pyplot

from torch.autograd import Variable

import time

from torchsummary import summary


train_data = VideoDataset(txt_path = '/home/juyoung0927/make_text_file/64_list/train/64_train_list.txt', root_dir='/projects/vode/team1/64_segment', num_channel = 3, clip_len=64)
train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=2)

valid_data = VideoDataset(txt_path = '/home/juyoung0927/make_text_file/64_list/val/64_val_list.txt', root_dir='/projects/vode/team1/64_segment', num_channel = 3, clip_len=64)
valid_loader = DataLoader(valid_data, batch_size=8, shuffle=False, num_workers=2)

trainval_loaders = {'train': train_loader, 'val': valid_loader}
trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}    

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print("Device being used:", device)

model = i3d.InceptionI3d(num_classes=157, in_channels=3) #class : normal, kidnap, vandalism, swoon, burglary, fight


weights = torch.load('./weightFile/rgb_charades.pt')
# weights = torch.load('./checkpoint/64-i3d-checkPoint.pt')
model.load_state_dict(weights, strict=False)

model.replace_logits(7)
#i3d.load_state_dict(torch.load('/ssd/models/000920.pt'))
# model.cuda()
# print(model.state_dict().keys())

for para in model.parameters():
    para.requires_grad = False

for name, param in model.named_parameters():
    if name in ['Mixed_4b.b0.conv3d.weight', 'Mixed_4b.b0.bn.weight', 'Mixed_4b.b0.bn.bias', 'Mixed_4b.b0.bn.running_mean', 'Mixed_4b.b0.bn.running_var', 'Mixed_4b.b0.bn.num_batches_tracked', 'Mixed_4b.b1a.conv3d.weight', 'Mixed_4b.b1a.bn.weight', 'Mixed_4b.b1a.bn.bias', 'Mixed_4b.b1a.bn.running_mean', 'Mixed_4b.b1a.bn.running_var', 'Mixed_4b.b1a.bn.num_batches_tracked', 'Mixed_4b.b1b.conv3d.weight', 'Mixed_4b.b1b.bn.weight', 'Mixed_4b.b1b.bn.bias', 'Mixed_4b.b1b.bn.running_mean', 'Mixed_4b.b1b.bn.running_var', 'Mixed_4b.b1b.bn.num_batches_tracked', 'Mixed_4b.b2a.conv3d.weight', 'Mixed_4b.b2a.bn.weight', 'Mixed_4b.b2a.bn.bias', 'Mixed_4b.b2a.bn.running_mean', 'Mixed_4b.b2a.bn.running_var', 'Mixed_4b.b2a.bn.num_batches_tracked', 'Mixed_4b.b2b.conv3d.weight', 'Mixed_4b.b2b.bn.weight', 'Mixed_4b.b2b.bn.bias', 'Mixed_4b.b2b.bn.running_mean', 'Mixed_4b.b2b.bn.running_var', 'Mixed_4b.b2b.bn.num_batches_tracked', 'Mixed_4b.b3b.conv3d.weight', 'Mixed_4b.b3b.bn.weight', 'Mixed_4b.b3b.bn.bias', 'Mixed_4b.b3b.bn.running_mean', 'Mixed_4b.b3b.bn.running_var', 'Mixed_4b.b3b.bn.num_batches_tracked', 'Mixed_4c.b0.conv3d.weight', 'Mixed_4c.b0.bn.weight', 'Mixed_4c.b0.bn.bias', 'Mixed_4c.b0.bn.running_mean', 'Mixed_4c.b0.bn.running_var', 'Mixed_4c.b0.bn.num_batches_tracked', 'Mixed_4c.b1a.conv3d.weight', 'Mixed_4c.b1a.bn.weight', 'Mixed_4c.b1a.bn.bias', 'Mixed_4c.b1a.bn.running_mean', 'Mixed_4c.b1a.bn.running_var', 'Mixed_4c.b1a.bn.num_batches_tracked', 'Mixed_4c.b1b.conv3d.weight', 'Mixed_4c.b1b.bn.weight', 'Mixed_4c.b1b.bn.bias', 'Mixed_4c.b1b.bn.running_mean', 'Mixed_4c.b1b.bn.running_var', 'Mixed_4c.b1b.bn.num_batches_tracked', 'Mixed_4c.b2a.conv3d.weight', 'Mixed_4c.b2a.bn.weight', 'Mixed_4c.b2a.bn.bias', 'Mixed_4c.b2a.bn.running_mean', 'Mixed_4c.b2a.bn.running_var', 'Mixed_4c.b2a.bn.num_batches_tracked', 'Mixed_4c.b2b.conv3d.weight', 'Mixed_4c.b2b.bn.weight', 'Mixed_4c.b2b.bn.bias', 'Mixed_4c.b2b.bn.running_mean', 'Mixed_4c.b2b.bn.running_var', 'Mixed_4c.b2b.bn.num_batches_tracked', 'Mixed_4c.b3b.conv3d.weight', 'Mixed_4c.b3b.bn.weight', 'Mixed_4c.b3b.bn.bias', 'Mixed_4c.b3b.bn.running_mean', 'Mixed_4c.b3b.bn.running_var', 'Mixed_4c.b3b.bn.num_batches_tracked', 'Mixed_4d.b0.conv3d.weight', 'Mixed_4d.b0.bn.weight', 'Mixed_4d.b0.bn.bias', 'Mixed_4d.b0.bn.running_mean', 'Mixed_4d.b0.bn.running_var', 'Mixed_4d.b0.bn.num_batches_tracked', 'Mixed_4d.b1a.conv3d.weight', 'Mixed_4d.b1a.bn.weight', 'Mixed_4d.b1a.bn.bias', 'Mixed_4d.b1a.bn.running_mean', 'Mixed_4d.b1a.bn.running_var', 'Mixed_4d.b1a.bn.num_batches_tracked', 'Mixed_4d.b1b.conv3d.weight', 'Mixed_4d.b1b.bn.weight', 'Mixed_4d.b1b.bn.bias', 'Mixed_4d.b1b.bn.running_mean', 'Mixed_4d.b1b.bn.running_var', 'Mixed_4d.b1b.bn.num_batches_tracked', 'Mixed_4d.b2a.conv3d.weight', 'Mixed_4d.b2a.bn.weight', 'Mixed_4d.b2a.bn.bias', 'Mixed_4d.b2a.bn.running_mean', 'Mixed_4d.b2a.bn.running_var', 'Mixed_4d.b2a.bn.num_batches_tracked', 'Mixed_4d.b2b.conv3d.weight', 'Mixed_4d.b2b.bn.weight', 'Mixed_4d.b2b.bn.bias', 'Mixed_4d.b2b.bn.running_mean', 'Mixed_4d.b2b.bn.running_var', 'Mixed_4d.b2b.bn.num_batches_tracked', 'Mixed_4d.b3b.conv3d.weight', 'Mixed_4d.b3b.bn.weight', 'Mixed_4d.b3b.bn.bias', 'Mixed_4d.b3b.bn.running_mean', 'Mixed_4d.b3b.bn.running_var', 'Mixed_4d.b3b.bn.num_batches_tracked', 'Mixed_4e.b0.conv3d.weight', 'Mixed_4e.b0.bn.weight', 'Mixed_4e.b0.bn.bias', 'Mixed_4e.b0.bn.running_mean', 'Mixed_4e.b0.bn.running_var', 'Mixed_4e.b0.bn.num_batches_tracked', 'Mixed_4e.b1a.conv3d.weight', 'Mixed_4e.b1a.bn.weight', 'Mixed_4e.b1a.bn.bias', 'Mixed_4e.b1a.bn.running_mean', 'Mixed_4e.b1a.bn.running_var', 'Mixed_4e.b1a.bn.num_batches_tracked', 'Mixed_4e.b1b.conv3d.weight', 'Mixed_4e.b1b.bn.weight', 'Mixed_4e.b1b.bn.bias', 'Mixed_4e.b1b.bn.running_mean', 'Mixed_4e.b1b.bn.running_var', 'Mixed_4e.b1b.bn.num_batches_tracked', 'Mixed_4e.b2a.conv3d.weight', 'Mixed_4e.b2a.bn.weight', 'Mixed_4e.b2a.bn.bias', 'Mixed_4e.b2a.bn.running_mean', 'Mixed_4e.b2a.bn.running_var', 'Mixed_4e.b2a.bn.num_batches_tracked', 'Mixed_4e.b2b.conv3d.weight', 'Mixed_4e.b2b.bn.weight', 'Mixed_4e.b2b.bn.bias', 'Mixed_4e.b2b.bn.running_mean', 'Mixed_4e.b2b.bn.running_var', 'Mixed_4e.b2b.bn.num_batches_tracked', 'Mixed_4e.b3b.conv3d.weight', 'Mixed_4e.b3b.bn.weight', 'Mixed_4e.b3b.bn.bias', 'Mixed_4e.b3b.bn.running_mean', 'Mixed_4e.b3b.bn.running_var', 'Mixed_4e.b3b.bn.num_batches_tracked', 'Mixed_4f.b0.conv3d.weight', 'Mixed_4f.b0.bn.weight', 'Mixed_4f.b0.bn.bias', 'Mixed_4f.b0.bn.running_mean', 'Mixed_4f.b0.bn.running_var', 'Mixed_4f.b0.bn.num_batches_tracked', 'Mixed_4f.b1a.conv3d.weight', 'Mixed_4f.b1a.bn.weight', 'Mixed_4f.b1a.bn.bias', 'Mixed_4f.b1a.bn.running_mean', 'Mixed_4f.b1a.bn.running_var', 'Mixed_4f.b1a.bn.num_batches_tracked', 'Mixed_4f.b1b.conv3d.weight', 'Mixed_4f.b1b.bn.weight', 'Mixed_4f.b1b.bn.bias', 'Mixed_4f.b1b.bn.running_mean', 'Mixed_4f.b1b.bn.running_var', 'Mixed_4f.b1b.bn.num_batches_tracked', 'Mixed_4f.b2a.conv3d.weight', 'Mixed_4f.b2a.bn.weight', 'Mixed_4f.b2a.bn.bias', 'Mixed_4f.b2a.bn.running_mean', 'Mixed_4f.b2a.bn.running_var', 'Mixed_4f.b2a.bn.num_batches_tracked', 'Mixed_4f.b2b.conv3d.weight', 'Mixed_4f.b2b.bn.weight', 'Mixed_4f.b2b.bn.bias', 'Mixed_4f.b2b.bn.running_mean', 'Mixed_4f.b2b.bn.running_var', 'Mixed_4f.b2b.bn.num_batches_tracked', 'Mixed_4f.b3b.conv3d.weight', 'Mixed_4f.b3b.bn.weight', 'Mixed_4f.b3b.bn.bias', 'Mixed_4f.b3b.bn.running_mean', 'Mixed_4f.b3b.bn.running_var', 'Mixed_4f.b3b.bn.num_batches_tracked', 'Mixed_5b.b0.conv3d.weight', 'Mixed_5b.b0.bn.weight', 'Mixed_5b.b0.bn.bias', 'Mixed_5b.b0.bn.running_mean', 'Mixed_5b.b0.bn.running_var', 'Mixed_5b.b0.bn.num_batches_tracked', 'Mixed_5b.b1a.conv3d.weight', 'Mixed_5b.b1a.bn.weight', 'Mixed_5b.b1a.bn.bias', 'Mixed_5b.b1a.bn.running_mean', 'Mixed_5b.b1a.bn.running_var', 'Mixed_5b.b1a.bn.num_batches_tracked', 'Mixed_5b.b1b.conv3d.weight', 'Mixed_5b.b1b.bn.weight', 'Mixed_5b.b1b.bn.bias', 'Mixed_5b.b1b.bn.running_mean', 'Mixed_5b.b1b.bn.running_var', 'Mixed_5b.b1b.bn.num_batches_tracked', 'Mixed_5b.b2a.conv3d.weight', 'Mixed_5b.b2a.bn.weight', 'Mixed_5b.b2a.bn.bias', 'Mixed_5b.b2a.bn.running_mean', 'Mixed_5b.b2a.bn.running_var', 'Mixed_5b.b2a.bn.num_batches_tracked', 'Mixed_5b.b2b.conv3d.weight', 'Mixed_5b.b2b.bn.weight', 'Mixed_5b.b2b.bn.bias', 'Mixed_5b.b2b.bn.running_mean', 'Mixed_5b.b2b.bn.running_var', 'Mixed_5b.b2b.bn.num_batches_tracked', 'Mixed_5b.b3b.conv3d.weight', 'Mixed_5b.b3b.bn.weight', 'Mixed_5b.b3b.bn.bias', 'Mixed_5b.b3b.bn.running_mean', 'Mixed_5b.b3b.bn.running_var', 'Mixed_5b.b3b.bn.num_batches_tracked', 'Mixed_5c.b0.conv3d.weight', 'Mixed_5c.b0.bn.weight', 'Mixed_5c.b0.bn.bias', 'Mixed_5c.b0.bn.running_mean', 'Mixed_5c.b0.bn.running_var', 'Mixed_5c.b0.bn.num_batches_tracked', 'Mixed_5c.b1a.conv3d.weight', 'Mixed_5c.b1a.bn.weight', 'Mixed_5c.b1a.bn.bias', 'Mixed_5c.b1a.bn.running_mean', 'Mixed_5c.b1a.bn.running_var', 'Mixed_5c.b1a.bn.num_batches_tracked', 'Mixed_5c.b1b.conv3d.weight', 'Mixed_5c.b1b.bn.weight', 'Mixed_5c.b1b.bn.bias', 'Mixed_5c.b1b.bn.running_mean', 'Mixed_5c.b1b.bn.running_var', 'Mixed_5c.b1b.bn.num_batches_tracked', 'Mixed_5c.b2a.conv3d.weight', 'Mixed_5c.b2a.bn.weight', 'Mixed_5c.b2a.bn.bias', 'Mixed_5c.b2a.bn.running_mean', 'Mixed_5c.b2a.bn.running_var', 'Mixed_5c.b2a.bn.num_batches_tracked', 'Mixed_5c.b2b.conv3d.weight', 'Mixed_5c.b2b.bn.weight', 'Mixed_5c.b2b.bn.bias', 'Mixed_5c.b2b.bn.running_mean', 'Mixed_5c.b2b.bn.running_var', 'Mixed_5c.b2b.bn.num_batches_tracked', 'Mixed_5c.b3b.conv3d.weight', 'Mixed_5c.b3b.bn.weight', 'Mixed_5c.b3b.bn.bias', 'Mixed_5c.b3b.bn.running_mean', 'Mixed_5c.b3b.bn.running_var', 'Mixed_5c.b3b.bn.num_batches_tracked']:
        param.requires_grad = True

    
LEARNING_RATE = 0.01
NUM_EPOCHS = 10000
WEIGHT_DECAY = 0.1
# BATCH_SIZE = params['batch_size']
model_path = './checkpoint2/64-16-i3d-checkPoint.pt'

# Architecture
NUM_CLASSES = 7 # FOR VIOLENCE DETECTION (BIONARY CIASSIFICATION PROBLEM)
# DEVICE = 'cuda:1'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# train_params = [{'params': c3d.get_1x_lr_params(model), 'lr': LEARNING_RATE},
#                         {'params': c3d.get_10x_lr_params(model), 'lr': LEARNING_RATE}]

# optimizer = torch.optim.SGD(train_params, lr=LEARNING_RATE)

# optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE) #기존
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0000001)   #추가
criterion = torch.nn.CrossEntropyLoss()
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10,gamma=0.1)    #추가
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [100, 200])   #추가 #기존 300,1000

best_acc = 0.0
valid_acc = 0.0
save_loss = []
save_loss_val = []
save_acc = []
save_acc_val = []


for epoch in range(NUM_EPOCHS):

    print("***************epoch : ", epoch+1)
    for phase in ['train', 'val']:

        running_loss = 0.0
        running_corrects = 0.0

        if phase == 'train':
            model.train()
        else:
            model.eval()

        correct_pred, num_examples = 0.0, 0
        for input, labels, name in trainval_loaders[phase]:
            #inputs, labels, path = data
            #print("1. labels-----", labels)

            optimizer.zero_grad() 

            input = input.to(device)
            # print("1. inputs ----", input.shape)

            if phase == 'train':
                outputs = model(input)
            else:
                with torch.no_grad():
                    outputs = model(input)

            # print("2.1. outputs(squeeze 전) ----", outputs.shape)
            outputs = outputs.squeeze(dim=2) # 추가함
            outputs = torch.nn.functional.softmax(outputs, dim=0)
            # print("-----------here----------", outputs.shape)
            outputs = outputs.cpu()
            # print("2. outputs ----", outputs)
            # print("2. outputs ----", outputs.shape)
            pred = torch.max(outputs,1)[1]
            # print("3. pred  -----", pred.shape)
            loss = criterion(outputs, labels)


            if phase == 'train':
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()   #추가
                scheduler.step() #추가

            running_loss += loss.item() * input.size(0)
            running_corrects += torch.sum(pred == labels.data)


            num_examples += labels.size(0)
            correct_pred += (pred == labels).sum()

        epoch_loss = running_loss / trainval_sizes[phase]
        #saveloss.append(epoch_loss)
        epoch_acc = running_corrects.double() / trainval_sizes[phase]

        if phase == 'train':
            print('train epoch_loss', epoch_loss)
            save_loss.append(epoch_loss)
            print('train epoch_acc', epoch_acc,'\n')
            save_acc.append(epoch_acc)
        else:
            print('valid epoch_loss', epoch_loss)
            save_loss_val.append(epoch_loss)
            print('valid epoch_acc', epoch_acc,'\n\n')
            save_acc_val.append(epoch_acc)
            valid_acc = epoch_acc
        
        if epoch%20 == 0:
            file_name = '/projects/vode/juyoung/I3D_unlabel_unlabelweight_control2/model/model_epoch_'+str(epoch+1)+'.pt'
            torch.save(model.state_dict(), file_name)
            pyplot.plot(save_loss, label='train')
            pyplot.plot(save_loss_val, label='validation')  
            pyplot.legend() 
            pyplot.title('loss')
            pyplot.xlabel('epoch')
            loss_file_name = '/projects/vode/juyoung/I3D_unlabel_unlabelweight_control2/loss/loss_epoch_'+str(epoch+1)+'.png'
            pyplot.savefig(loss_file_name)
            pyplot.clf()
            pyplot.plot(save_acc, label='train')
            pyplot.plot(save_acc_val, label='validation')  
            pyplot.legend() 
            pyplot.title('accuracy')
            pyplot.xlabel('epoch')
            acc_file_name = '/projects/vode/juyoung/I3D_unlabel_unlabelweight_control2/accuracy/acc_epoch_'+str(epoch+1)+'.png'
            pyplot.savefig(acc_file_name)
            pyplot.clf()

        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), model_path)
            print('best_acc at Epoch: ', epoch+1)
            print('saving model with acc {:.3f}'.format(best_acc))

