import torch
import torch.optim as optim
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


train_data = VideoDataset(txt_path = '/home/juyoung0927/custom_I3D/list_txt/train/all_train_list.txt', root_dir='/projects/vode/team1/I3D', num_channel = 3, clip_len=64)
train_loader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=1)

valid_data = VideoDataset(txt_path = '/home/juyoung0927/custom_I3D/list_txt/val/all_val_list.txt', root_dir='/projects/vode/team1/I3D', num_channel = 3, clip_len=64)
valid_loader = DataLoader(valid_data, batch_size=4, shuffle=False, num_workers=1)
    

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print("Device being used:", device)

model = i3d.InceptionI3d(num_classes=157, in_channels=3) #class : normal, kidnap, vandalism, swoon, burglary, fight
# model = c3d.P3D(num_classes=6, pretrained=True) #class : normal, kidnap, vandalism, swoon, burglary, fight
# model = c3d.C3D(num_classes=6, pretrained=True) #class : normal, kidnap, vandalism, swoon, burglary, fight

# summary(model, input_size=(3, 224, 224, 224))

# print(model.state_dict().keys())

weights = torch.load('./weightFile/rgb_charades.pt')
model.load_state_dict(weights, strict=False)

model.replace_logits(6)
#i3d.load_state_dict(torch.load('/ssd/models/000920.pt'))
# model.cuda()

for para in model.parameters():
    para.requires_grad = False

for name, param in model.named_parameters():
    if name in ['Mixed_4b.b0.conv3d.weight', 'Mixed_4b.b0.bn.weight', 'Mixed_4b.b0.bn.bias', 'Mixed_4b.b0.bn.running_mean', 'Mixed_4b.b0.bn.running_var', 'Mixed_4b.b0.bn.num_batches_tracked', 'Mixed_4b.b1a.conv3d.weight', 'Mixed_4b.b1a.bn.weight', 'Mixed_4b.b1a.bn.bias', 'Mixed_4b.b1a.bn.running_mean', 'Mixed_4b.b1a.bn.running_var', 'Mixed_4b.b1a.bn.num_batches_tracked', 'Mixed_4b.b1b.conv3d.weight', 'Mixed_4b.b1b.bn.weight', 'Mixed_4b.b1b.bn.bias', 'Mixed_4b.b1b.bn.running_mean', 'Mixed_4b.b1b.bn.running_var', 'Mixed_4b.b1b.bn.num_batches_tracked', 'Mixed_4b.b2a.conv3d.weight', 'Mixed_4b.b2a.bn.weight', 'Mixed_4b.b2a.bn.bias', 'Mixed_4b.b2a.bn.running_mean', 'Mixed_4b.b2a.bn.running_var', 'Mixed_4b.b2a.bn.num_batches_tracked', 'Mixed_4b.b2b.conv3d.weight', 'Mixed_4b.b2b.bn.weight', 'Mixed_4b.b2b.bn.bias', 'Mixed_4b.b2b.bn.running_mean', 'Mixed_4b.b2b.bn.running_var', 'Mixed_4b.b2b.bn.num_batches_tracked', 'Mixed_4b.b3b.conv3d.weight', 'Mixed_4b.b3b.bn.weight', 'Mixed_4b.b3b.bn.bias', 'Mixed_4b.b3b.bn.running_mean', 'Mixed_4b.b3b.bn.running_var', 'Mixed_4b.b3b.bn.num_batches_tracked', 'Mixed_4c.b0.conv3d.weight', 'Mixed_4c.b0.bn.weight', 'Mixed_4c.b0.bn.bias', 'Mixed_4c.b0.bn.running_mean', 'Mixed_4c.b0.bn.running_var', 'Mixed_4c.b0.bn.num_batches_tracked', 'Mixed_4c.b1a.conv3d.weight', 'Mixed_4c.b1a.bn.weight', 'Mixed_4c.b1a.bn.bias', 'Mixed_4c.b1a.bn.running_mean', 'Mixed_4c.b1a.bn.running_var', 'Mixed_4c.b1a.bn.num_batches_tracked', 'Mixed_4c.b1b.conv3d.weight', 'Mixed_4c.b1b.bn.weight', 'Mixed_4c.b1b.bn.bias', 'Mixed_4c.b1b.bn.running_mean', 'Mixed_4c.b1b.bn.running_var', 'Mixed_4c.b1b.bn.num_batches_tracked', 'Mixed_4c.b2a.conv3d.weight', 'Mixed_4c.b2a.bn.weight', 'Mixed_4c.b2a.bn.bias', 'Mixed_4c.b2a.bn.running_mean', 'Mixed_4c.b2a.bn.running_var', 'Mixed_4c.b2a.bn.num_batches_tracked', 'Mixed_4c.b2b.conv3d.weight', 'Mixed_4c.b2b.bn.weight', 'Mixed_4c.b2b.bn.bias', 'Mixed_4c.b2b.bn.running_mean', 'Mixed_4c.b2b.bn.running_var', 'Mixed_4c.b2b.bn.num_batches_tracked', 'Mixed_4c.b3b.conv3d.weight', 'Mixed_4c.b3b.bn.weight', 'Mixed_4c.b3b.bn.bias', 'Mixed_4c.b3b.bn.running_mean', 'Mixed_4c.b3b.bn.running_var', 'Mixed_4c.b3b.bn.num_batches_tracked', 'Mixed_4d.b0.conv3d.weight', 'Mixed_4d.b0.bn.weight', 'Mixed_4d.b0.bn.bias', 'Mixed_4d.b0.bn.running_mean', 'Mixed_4d.b0.bn.running_var', 'Mixed_4d.b0.bn.num_batches_tracked', 'Mixed_4d.b1a.conv3d.weight', 'Mixed_4d.b1a.bn.weight', 'Mixed_4d.b1a.bn.bias', 'Mixed_4d.b1a.bn.running_mean', 'Mixed_4d.b1a.bn.running_var', 'Mixed_4d.b1a.bn.num_batches_tracked', 'Mixed_4d.b1b.conv3d.weight', 'Mixed_4d.b1b.bn.weight', 'Mixed_4d.b1b.bn.bias', 'Mixed_4d.b1b.bn.running_mean', 'Mixed_4d.b1b.bn.running_var', 'Mixed_4d.b1b.bn.num_batches_tracked', 'Mixed_4d.b2a.conv3d.weight', 'Mixed_4d.b2a.bn.weight', 'Mixed_4d.b2a.bn.bias', 'Mixed_4d.b2a.bn.running_mean', 'Mixed_4d.b2a.bn.running_var', 'Mixed_4d.b2a.bn.num_batches_tracked', 'Mixed_4d.b2b.conv3d.weight', 'Mixed_4d.b2b.bn.weight', 'Mixed_4d.b2b.bn.bias', 'Mixed_4d.b2b.bn.running_mean', 'Mixed_4d.b2b.bn.running_var', 'Mixed_4d.b2b.bn.num_batches_tracked', 'Mixed_4d.b3b.conv3d.weight', 'Mixed_4d.b3b.bn.weight', 'Mixed_4d.b3b.bn.bias', 'Mixed_4d.b3b.bn.running_mean', 'Mixed_4d.b3b.bn.running_var', 'Mixed_4d.b3b.bn.num_batches_tracked', 'Mixed_4e.b0.conv3d.weight', 'Mixed_4e.b0.bn.weight', 'Mixed_4e.b0.bn.bias', 'Mixed_4e.b0.bn.running_mean', 'Mixed_4e.b0.bn.running_var', 'Mixed_4e.b0.bn.num_batches_tracked', 'Mixed_4e.b1a.conv3d.weight', 'Mixed_4e.b1a.bn.weight', 'Mixed_4e.b1a.bn.bias', 'Mixed_4e.b1a.bn.running_mean', 'Mixed_4e.b1a.bn.running_var', 'Mixed_4e.b1a.bn.num_batches_tracked', 'Mixed_4e.b1b.conv3d.weight', 'Mixed_4e.b1b.bn.weight', 'Mixed_4e.b1b.bn.bias', 'Mixed_4e.b1b.bn.running_mean', 'Mixed_4e.b1b.bn.running_var', 'Mixed_4e.b1b.bn.num_batches_tracked', 'Mixed_4e.b2a.conv3d.weight', 'Mixed_4e.b2a.bn.weight', 'Mixed_4e.b2a.bn.bias', 'Mixed_4e.b2a.bn.running_mean', 'Mixed_4e.b2a.bn.running_var', 'Mixed_4e.b2a.bn.num_batches_tracked', 'Mixed_4e.b2b.conv3d.weight', 'Mixed_4e.b2b.bn.weight', 'Mixed_4e.b2b.bn.bias', 'Mixed_4e.b2b.bn.running_mean', 'Mixed_4e.b2b.bn.running_var', 'Mixed_4e.b2b.bn.num_batches_tracked', 'Mixed_4e.b3b.conv3d.weight', 'Mixed_4e.b3b.bn.weight', 'Mixed_4e.b3b.bn.bias', 'Mixed_4e.b3b.bn.running_mean', 'Mixed_4e.b3b.bn.running_var', 'Mixed_4e.b3b.bn.num_batches_tracked', 'Mixed_4f.b0.conv3d.weight', 'Mixed_4f.b0.bn.weight', 'Mixed_4f.b0.bn.bias', 'Mixed_4f.b0.bn.running_mean', 'Mixed_4f.b0.bn.running_var', 'Mixed_4f.b0.bn.num_batches_tracked', 'Mixed_4f.b1a.conv3d.weight', 'Mixed_4f.b1a.bn.weight', 'Mixed_4f.b1a.bn.bias', 'Mixed_4f.b1a.bn.running_mean', 'Mixed_4f.b1a.bn.running_var', 'Mixed_4f.b1a.bn.num_batches_tracked', 'Mixed_4f.b1b.conv3d.weight', 'Mixed_4f.b1b.bn.weight', 'Mixed_4f.b1b.bn.bias', 'Mixed_4f.b1b.bn.running_mean', 'Mixed_4f.b1b.bn.running_var', 'Mixed_4f.b1b.bn.num_batches_tracked', 'Mixed_4f.b2a.conv3d.weight', 'Mixed_4f.b2a.bn.weight', 'Mixed_4f.b2a.bn.bias', 'Mixed_4f.b2a.bn.running_mean', 'Mixed_4f.b2a.bn.running_var', 'Mixed_4f.b2a.bn.num_batches_tracked', 'Mixed_4f.b2b.conv3d.weight', 'Mixed_4f.b2b.bn.weight', 'Mixed_4f.b2b.bn.bias', 'Mixed_4f.b2b.bn.running_mean', 'Mixed_4f.b2b.bn.running_var', 'Mixed_4f.b2b.bn.num_batches_tracked', 'Mixed_4f.b3b.conv3d.weight', 'Mixed_4f.b3b.bn.weight', 'Mixed_4f.b3b.bn.bias', 'Mixed_4f.b3b.bn.running_mean', 'Mixed_4f.b3b.bn.running_var', 'Mixed_4f.b3b.bn.num_batches_tracked', 'Mixed_5b.b0.conv3d.weight', 'Mixed_5b.b0.bn.weight', 'Mixed_5b.b0.bn.bias', 'Mixed_5b.b0.bn.running_mean', 'Mixed_5b.b0.bn.running_var', 'Mixed_5b.b0.bn.num_batches_tracked', 'Mixed_5b.b1a.conv3d.weight', 'Mixed_5b.b1a.bn.weight', 'Mixed_5b.b1a.bn.bias', 'Mixed_5b.b1a.bn.running_mean', 'Mixed_5b.b1a.bn.running_var', 'Mixed_5b.b1a.bn.num_batches_tracked', 'Mixed_5b.b1b.conv3d.weight', 'Mixed_5b.b1b.bn.weight', 'Mixed_5b.b1b.bn.bias', 'Mixed_5b.b1b.bn.running_mean', 'Mixed_5b.b1b.bn.running_var', 'Mixed_5b.b1b.bn.num_batches_tracked', 'Mixed_5b.b2a.conv3d.weight', 'Mixed_5b.b2a.bn.weight', 'Mixed_5b.b2a.bn.bias', 'Mixed_5b.b2a.bn.running_mean', 'Mixed_5b.b2a.bn.running_var', 'Mixed_5b.b2a.bn.num_batches_tracked', 'Mixed_5b.b2b.conv3d.weight', 'Mixed_5b.b2b.bn.weight', 'Mixed_5b.b2b.bn.bias', 'Mixed_5b.b2b.bn.running_mean', 'Mixed_5b.b2b.bn.running_var', 'Mixed_5b.b2b.bn.num_batches_tracked', 'Mixed_5b.b3b.conv3d.weight', 'Mixed_5b.b3b.bn.weight', 'Mixed_5b.b3b.bn.bias', 'Mixed_5b.b3b.bn.running_mean', 'Mixed_5b.b3b.bn.running_var', 'Mixed_5b.b3b.bn.num_batches_tracked', 'Mixed_5c.b0.conv3d.weight', 'Mixed_5c.b0.bn.weight', 'Mixed_5c.b0.bn.bias', 'Mixed_5c.b0.bn.running_mean', 'Mixed_5c.b0.bn.running_var', 'Mixed_5c.b0.bn.num_batches_tracked', 'Mixed_5c.b1a.conv3d.weight', 'Mixed_5c.b1a.bn.weight', 'Mixed_5c.b1a.bn.bias', 'Mixed_5c.b1a.bn.running_mean', 'Mixed_5c.b1a.bn.running_var', 'Mixed_5c.b1a.bn.num_batches_tracked', 'Mixed_5c.b1b.conv3d.weight', 'Mixed_5c.b1b.bn.weight', 'Mixed_5c.b1b.bn.bias', 'Mixed_5c.b1b.bn.running_mean', 'Mixed_5c.b1b.bn.running_var', 'Mixed_5c.b1b.bn.num_batches_tracked', 'Mixed_5c.b2a.conv3d.weight', 'Mixed_5c.b2a.bn.weight', 'Mixed_5c.b2a.bn.bias', 'Mixed_5c.b2a.bn.running_mean', 'Mixed_5c.b2a.bn.running_var', 'Mixed_5c.b2a.bn.num_batches_tracked', 'Mixed_5c.b2b.conv3d.weight', 'Mixed_5c.b2b.bn.weight', 'Mixed_5c.b2b.bn.bias', 'Mixed_5c.b2b.bn.running_mean', 'Mixed_5c.b2b.bn.running_var', 'Mixed_5c.b2b.bn.num_batches_tracked', 'Mixed_5c.b3b.conv3d.weight', 'Mixed_5c.b3b.bn.weight', 'Mixed_5c.b3b.bn.bias', 'Mixed_5c.b3b.bn.running_mean', 'Mixed_5c.b3b.bn.running_var', 'Mixed_5c.b3b.bn.num_batches_tracked']:
        param.requires_grad = True

    
LEARNING_RATE = 0.0001
NUM_EPOCHS = 10000
WEIGHT_DECAY = 0.1
# BATCH_SIZE = params['batch_size']
model_path = './checkpoint/i3d-checkPoint.pt'

# Architecture
NUM_CLASSES = 6 # FOR VIOLENCE DETECTION (BIONARY CIASSIFICATION PROBLEM)
# DEVICE = 'cuda:1'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# train_params = [{'params': c3d.get_1x_lr_params(model), 'lr': LEARNING_RATE},
#                         {'params': c3d.get_10x_lr_params(model), 'lr': LEARNING_RATE}]

# optimizer = torch.optim.SGD(train_params, lr=LEARNING_RATE)
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()

print("Preperation Done")

def compute_accuracy(model, data_loader):
    correct_pred, num_examples = 0, 0

    for i, data in enumerate(data_loader, 0):
        targets = i
        #print("epoch : ", epoch)
        inputs, labels, path = data
        # print("1. labels-----", labels)
        inputs = inputs.to(device)

        

        outputs = model(inputs)
        outputs = outputs.cpu()

        pred = torch.max(outputs,1)[1]
        # print("2. pred-----", pred)
        # print("3.")

        num_examples += labels.size(0)
        correct_pred += (pred == labels).sum()

    return correct_pred.float()/num_examples * 100


# %%

def compute_epoch_loss(model, data_loader):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            n = len(data_loader)
            inputs, labels, path = data
            #print("1. labels-----", labels)

            optimizer.zero_grad()

            inputs = inputs.to(device)

            outputs = model(inputs)
            outputs = torch.nn.functional.softmax(outputs, dim=1)
            outputs = outputs.cpu()
            #print("2. outputs ----", outputs)

            #print("3. pred  -----", pred)
            loss = criterion(outputs, labels)
            #print("4. loss  ----", loss,'\n')

            running_loss += loss.item()


        return running_loss / n
    

minibatch_cost, epoch_cost = [], []
all_train_acc, all_valid_acc = [], []

start_time = time.time()

best_acc = 0.0


n = len(train_loader)
saveloss = []

saveloss_val = []
print("train_loader len  :  ", n)
for epoch in range(NUM_EPOCHS):
    
    # train_acc = []
    # loss = []
    running_loss = 0.0
    model.train()
    print("epoch : ", epoch+1)

    for i, data in enumerate(train_loader, 0):
        inputs, labels, path = data
        print("1. labels-----", labels)
        #print("labels -----", labels)
        #print("path ----", path)
        
        #labels = labels.cpu()

        optimizer.zero_grad()

        inputs = inputs.to(device)


        outputs = model(inputs)
        outputs = outputs.squeeze() # 추가함
        print(outputs.shape)
        outputs = torch.nn.functional.softmax(outputs, dim=1)
        outputs = outputs.cpu()
        # print("2. outputs ----", outputs.shape)
        pred = torch.max(outputs,1)[1]
        print("3. pred  -----", pred)
        loss = criterion(outputs, labels)
        # loss = Variable(loss, requires_grad = True)
        print("4. loss  ----", loss,'\n')
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    saveloss.append(running_loss / n)
    print('**************************epoch : ', epoch+1, '**********************loss : ', running_loss/n, '\n\n\n')

    model.eval()
    with torch.set_grad_enabled(False): # save memory during inference
        train_acc = compute_accuracy(model, train_loader)
        valid_acc = compute_accuracy(model, valid_loader)


        val_loss = compute_epoch_loss(model, valid_loader)
        saveloss_val.append(val_loss)
        # train_acc = accuracy(outputs.data, targets, topk=(1, ))
        
        # print(train_acc) 
        
        # valid_acc = accuracy(outputs.data, targets, topk=(1, ))

        if epoch%100 == 0:
            file_name = '/home/juyoung0927/custom_I3D/model/model_epoch_'+str(epoch+1)+'.pt'
            torch.save(model.state_dict(), file_name)
            pyplot.plot(saveloss, label='train')
            pyplot.plot(saveloss_val, label='validation')   
            pyplot.legend()
            pyplot.title('loss')
            pyplot.xlabel('epoch')
            loss_file_name = '/home/juyoung0927/custom_I3D/loss/loss_epoch_'+str(epoch+1)+'.png'
            pyplot.savefig(loss_file_name)
            pyplot.clf()
            
        print('Epoch: %03d/%03d | Train: %.3f%% | Valid: %.3f%%' % (
              epoch+1, NUM_EPOCHS, train_acc, valid_acc))
        
        
        # all_train_acc.append(train_acc)
        # all_valid_acc.append(valid_acc)
        # cost = compute_epoch_loss(model, train_loader)
        # epoch_cost.append(cost)
        
    
            # if the model improves, save a checkpoint at this epoch
    if valid_acc > best_acc:
        best_acc = valid_acc
        torch.save(model.state_dict(), model_path)
        print('best_acc at Epoch: ', epoch+1)
        print('saving model with acc {:.3f}'.format(best_acc))
        
    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

# Call flush() method to make sure that all pending events have been written to disk.



    