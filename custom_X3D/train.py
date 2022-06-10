import time
import torch.nn as nn
import os
from pathlib import Path
import torch
from dataloader.dataset import VideoDataset
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
from torch.autograd import Variable
from matplotlib import pyplot
from tqdm import tqdm
from network import X3D as x3d
import pkbar


X3D_VERSION = 'M' # ['S', 'M', 'XL']

BS = 8
BS_UPSCALE = 16 # CHANGE WITH GPU AVAILABILITY
#INIT_LR = (1.6/1024)*(BS*BS_UPSCALE)
INIT_LR = 0.2
SCHEDULE_SCALE = 4
EPOCHS = (60000 * 1024 * 1.5)/220000 #(~420)

LONG_CYCLE = [8, 4, 2, 1]
LONG_CYCLE_LR_SCALE = [8, 0.5, 0.5, 0.5]
GPUS = 2
BASE_BS_PER_GPU = BS * BS_UPSCALE // GPUS # FOR SPLIT BN
CONST_BN_SIZE = 8


def run(init_lr=INIT_LR, warmup_steps=8000, max_epochs=10000, batch_size=BS*BS_UPSCALE):
    train_data = VideoDataset(txt_path = '/home/dahee333/custom_X3D/list_txt/train/16_train_list.txt', chrom_txt_path = '/home/dahee333/custom_X3D/list_txt/train/16_train_list_chro.txt', num_channel = 3, clip_len=16)
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=10)

    valid_data = VideoDataset(txt_path = '/home/dahee333/custom_X3D/list_txt/val/16_val_list.txt', chrom_txt_path = '/home/dahee333/custom_X3D/list_txt/val/16_val_list_chro.txt', num_channel = 3, clip_len=16)
    valid_loader = DataLoader(valid_data, batch_size=8, shuffle=False, num_workers=10)


    trainval_loaders = {'train': train_loader, 'val': valid_loader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}

   
   
    gamma_tau = {'S':6, 'M':5*2, 'XL':5}[X3D_VERSION] # 'M':5 FOR LONGER SCHEDULE, NUM OF GPUS INCREASE
    st_steps = 204000 #0 # FOR LR WARM-UP
    load_steps = 24000 #0 # FOR LOADING AND PRINT SCHEDULE
    steps = 204000 #0
    epochs = 0#118 #0
    num_steps_per_update = 1 # ACCUMULATE GRADIENT IF NEEDED
    cur_iterations = steps * num_steps_per_update
    iterations_per_epoch = trainval_sizes['train']//batch_size
    val_iterations_per_epoch = trainval_sizes['val']//batch_size
    max_steps = iterations_per_epoch * max_epochs

    last_long = -2



    save_model = './model_chkt/x3d_AIhub_'
    model_path = '/home/dahee333/custom_X3D/checkPoint/x3d-checkPoint'

    model = x3d.generate_model(X3D_VERSION, base_bn_splits=1)

    # load pre-trained weights 
    # load_ckpt = torch.load('./weightFile/x3d_charades_rgb_sgd_024000.pt')
    # model.load_state_dict(load_ckpt['model_state_dict'])

    # change the very last layer, then modify the output
    #model.fc2 = nn.Linear(in_features = 2048, out_features = 6)


    for para in model.parameters():
        para.requires_grad = False

    for name, param in model.named_parameters():
        if name in ['layer4.0.conv1.weight', 'layer4.0.bn1.weight', 'layer4.0.bn1.bias', 'layer4.0.bn1.bn.running_mean', 'layer4.0.bn1.bn.running_var', 'layer4.0.bn1.bn.num_batches_tracked', 'layer4.0.bn1.split_bn.running_mean', 'layer4.0.bn1.split_bn.running_var', 'layer4.0.bn1.split_bn.num_batches_tracked', 'layer4.0.conv2.weight', 'layer4.0.bn2.weight', 'layer4.0.bn2.bias', 'layer4.0.bn2.bn.running_mean', 'layer4.0.bn2.bn.running_var', 'layer4.0.bn2.bn.num_batches_tracked', 'layer4.0.bn2.split_bn.running_mean', 'layer4.0.bn2.split_bn.running_var', 'layer4.0.bn2.split_bn.num_batches_tracked', 'layer4.0.conv3.weight', 'layer4.0.bn3.weight', 'layer4.0.bn3.bias', 'layer4.0.bn3.bn.running_mean', 'layer4.0.bn3.bn.running_var', 'layer4.0.bn3.bn.num_batches_tracked', 'layer4.0.bn3.split_bn.running_mean', 'layer4.0.bn3.split_bn.running_var', 'layer4.0.bn3.split_bn.num_batches_tracked', 'layer4.0.fc1.weight', 'layer4.0.fc1.bias', 'layer4.0.fc2.weight', 'layer4.0.fc2.bias', 'layer4.0.downsample.0.weight', 'layer4.0.downsample.1.weight', 'layer4.0.downsample.1.bias', 'layer4.0.downsample.1.bn.running_mean', 'layer4.0.downsample.1.bn.running_var', 'layer4.0.downsample.1.bn.num_batches_tracked', 'layer4.0.downsample.1.split_bn.running_mean', 'layer4.0.downsample.1.split_bn.running_var', 'layer4.0.downsample.1.split_bn.num_batches_tracked', 'layer4.1.conv1.weight', 'layer4.1.bn1.weight', 'layer4.1.bn1.bias', 'layer4.1.bn1.bn.running_mean', 'layer4.1.bn1.bn.running_var', 'layer4.1.bn1.bn.num_batches_tracked', 'layer4.1.bn1.split_bn.running_mean', 'layer4.1.bn1.split_bn.running_var', 'layer4.1.bn1.split_bn.num_batches_tracked', 'layer4.1.conv2.weight', 'layer4.1.bn2.weight', 'layer4.1.bn2.bias', 'layer4.1.bn2.bn.running_mean', 'layer4.1.bn2.bn.running_var', 'layer4.1.bn2.bn.num_batches_tracked', 'layer4.1.bn2.split_bn.running_mean', 'layer4.1.bn2.split_bn.running_var', 'layer4.1.bn2.split_bn.num_batches_tracked', 'layer4.1.conv3.weight', 'layer4.1.bn3.weight', 'layer4.1.bn3.bias', 'layer4.1.bn3.bn.running_mean', 'layer4.1.bn3.bn.running_var', 'layer4.1.bn3.bn.num_batches_tracked', 'layer4.1.bn3.split_bn.running_mean', 'layer4.1.bn3.split_bn.running_var', 'layer4.1.bn3.split_bn.num_batches_tracked', 'layer4.2.conv1.weight', 'layer4.2.bn1.weight', 'layer4.2.bn1.bias', 'layer4.2.bn1.bn.running_mean', 'layer4.2.bn1.bn.running_var', 'layer4.2.bn1.bn.num_batches_tracked', 'layer4.2.bn1.split_bn.running_mean', 'layer4.2.bn1.split_bn.running_var', 'layer4.2.bn1.split_bn.num_batches_tracked', 'layer4.2.conv2.weight', 'layer4.2.bn2.weight', 'layer4.2.bn2.bias', 'layer4.2.bn2.bn.running_mean', 'layer4.2.bn2.bn.running_var', 'layer4.2.bn2.bn.num_batches_tracked', 'layer4.2.bn2.split_bn.running_mean', 'layer4.2.bn2.split_bn.running_var', 'layer4.2.bn2.split_bn.num_batches_tracked', 'layer4.2.conv3.weight', 'layer4.2.bn3.weight', 'layer4.2.bn3.bias', 'layer4.2.bn3.bn.running_mean', 'layer4.2.bn3.bn.running_var', 'layer4.2.bn3.bn.num_batches_tracked', 'layer4.2.bn3.split_bn.running_mean', 'layer4.2.bn3.split_bn.running_var', 'layer4.2.bn3.split_bn.num_batches_tracked', 'layer4.2.fc1.weight', 'layer4.2.fc1.bias', 'layer4.2.fc2.weight', 'layer4.2.fc2.bias', 'layer4.3.conv1.weight', 'layer4.3.bn1.weight', 'layer4.3.bn1.bias', 'layer4.3.bn1.bn.running_mean', 'layer4.3.bn1.bn.running_var', 'layer4.3.bn1.bn.num_batches_tracked', 'layer4.3.bn1.split_bn.running_mean', 'layer4.3.bn1.split_bn.running_var', 'layer4.3.bn1.split_bn.num_batches_tracked', 'layer4.3.conv2.weight', 'layer4.3.bn2.weight', 'layer4.3.bn2.bias', 'layer4.3.bn2.bn.running_mean', 'layer4.3.bn2.bn.running_var', 'layer4.3.bn2.bn.num_batches_tracked', 'layer4.3.bn2.split_bn.running_mean', 'layer4.3.bn2.split_bn.running_var', 'layer4.3.bn2.split_bn.num_batches_tracked', 'layer4.3.conv3.weight', 'layer4.3.bn3.weight', 'layer4.3.bn3.bias', 'layer4.3.bn3.bn.running_mean', 'layer4.3.bn3.bn.running_var', 'layer4.3.bn3.bn.num_batches_tracked', 'layer4.3.bn3.split_bn.running_mean', 'layer4.3.bn3.split_bn.running_var', 'layer4.3.bn3.split_bn.num_batches_tracked', 'layer4.4.conv1.weight', 'layer4.4.bn1.weight', 'layer4.4.bn1.bias', 'layer4.4.bn1.bn.running_mean', 'layer4.4.bn1.bn.running_var', 'layer4.4.bn1.bn.num_batches_tracked', 'layer4.4.bn1.split_bn.running_mean', 'layer4.4.bn1.split_bn.running_var', 'layer4.4.bn1.split_bn.num_batches_tracked', 'layer4.4.conv2.weight', 'layer4.4.bn2.weight', 'layer4.4.bn2.bias', 'layer4.4.bn2.bn.running_mean', 'layer4.4.bn2.bn.running_var', 'layer4.4.bn2.bn.num_batches_tracked', 'layer4.4.bn2.split_bn.running_mean', 'layer4.4.bn2.split_bn.running_var', 'layer4.4.bn2.split_bn.num_batches_tracked', 'layer4.4.conv3.weight', 'layer4.4.bn3.weight', 'layer4.4.bn3.bias', 'layer4.4.bn3.bn.running_mean', 'layer4.4.bn3.bn.running_var', 'layer4.4.bn3.bn.num_batches_tracked', 'layer4.4.bn3.split_bn.running_mean', 'layer4.4.bn3.split_bn.running_var', 'layer4.4.bn3.split_bn.num_batches_tracked', 'layer4.4.fc1.weight', 'layer4.4.fc1.bias', 'layer4.4.fc2.weight', 'layer4.4.fc2.bias', 'layer4.5.conv1.weight', 'layer4.5.bn1.weight', 'layer4.5.bn1.bias', 'layer4.5.bn1.bn.running_mean', 'layer4.5.bn1.bn.running_var', 'layer4.5.bn1.bn.num_batches_tracked', 'layer4.5.bn1.split_bn.running_mean', 'layer4.5.bn1.split_bn.running_var', 'layer4.5.bn1.split_bn.num_batches_tracked', 'layer4.5.conv2.weight', 'layer4.5.bn2.weight', 'layer4.5.bn2.bias', 'layer4.5.bn2.bn.running_mean', 'layer4.5.bn2.bn.running_var', 'layer4.5.bn2.bn.num_batches_tracked', 'layer4.5.bn2.split_bn.running_mean', 'layer4.5.bn2.split_bn.running_var', 'layer4.5.bn2.split_bn.num_batches_tracked', 'layer4.5.conv3.weight', 'layer4.5.bn3.weight', 'layer4.5.bn3.bias', 'layer4.5.bn3.bn.running_mean', 'layer4.5.bn3.bn.running_var', 'layer4.5.bn3.bn.num_batches_tracked', 'layer4.5.bn3.split_bn.running_mean', 'layer4.5.bn3.split_bn.running_var', 'layer4.5.bn3.split_bn.num_batches_tracked', 'layer4.6.conv1.weight', 'layer4.6.bn1.weight', 'layer4.6.bn1.bias', 'layer4.6.bn1.bn.running_mean', 'layer4.6.bn1.bn.running_var', 'layer4.6.bn1.bn.num_batches_tracked', 'layer4.6.bn1.split_bn.running_mean', 'layer4.6.bn1.split_bn.running_var', 'layer4.6.bn1.split_bn.num_batches_tracked', 'layer4.6.conv2.weight', 'layer4.6.bn2.weight', 'layer4.6.bn2.bias', 'layer4.6.bn2.bn.running_mean', 'layer4.6.bn2.bn.running_var', 'layer4.6.bn2.bn.num_batches_tracked', 'layer4.6.bn2.split_bn.running_mean', 'layer4.6.bn2.split_bn.running_var', 'layer4.6.bn2.split_bn.num_batches_tracked', 'layer4.6.conv3.weight', 'layer4.6.bn3.weight', 'layer4.6.bn3.bias', 'layer4.6.bn3.bn.running_mean', 'layer4.6.bn3.bn.running_var', 'layer4.6.bn3.bn.num_batches_tracked', 'layer4.6.bn3.split_bn.running_mean', 'layer4.6.bn3.split_bn.running_var', 'layer4.6.bn3.split_bn.num_batches_tracked', 'layer4.6.fc1.weight', 'layer4.6.fc1.bias', 'layer4.6.fc2.weight', 'layer4.6.fc2.bias', 'conv5.weight', 'bn5.weight', 'bn5.bias', 'bn5.bn.running_mean', 'bn5.bn.running_var', 'bn5.bn.num_batches_tracked', 'bn5.split_bn.running_mean', 'bn5.split_bn.running_var', 'bn5.split_bn.num_batches_tracked', 'fc1.weight', 'fc2.weight', 'fc2.bias']:
            param.requires_grad = True


    RESTART = False
    if steps>0:
        load_ckpt = torch.load('./weightFile/x3d_charades_rgb_sgd_'+str(load_steps).zfill(6)+'.pt')
        #cur_long_ind = load_ckpt['long_ind']
        #bn_splits = model.update_bn_splits_long_cycle(LONG_CYCLE[cur_long_ind])

        model.load_state_dict(load_ckpt['model_state_dict'])
        model.fc2 = nn.Linear(in_features = 2048, out_features = 6)

        #last_long = cur_long_ind
        RESTART = True


    model.cuda()
    model = nn.DataParallel(model)
    print('model loaded')

    lr = init_lr
    print ('INIT LR: %f'%lr)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-5)
    #.MultiStepLR(optimizer, lr_schedule)
    # if steps>0:
    #     optimizer.load_state_dict(load_ckpt['optimizer_state_dict'])
        #lr_sched.load_state_dict(load_ckpt['scheduler_state_dict'])

    criterion = torch.nn.CrossEntropyLoss()



    best_acc = 0.0
    valid_acc = 0.0
    save_loss = []
    save_loss_val = []
    save_acc = []
    save_acc_val = []


    while epochs < max_epochs:
        print ('Step {} Epoch {}'.format(steps, epochs))
        print ('-' * 10)

        # Each epoch has a training and validation phase
        for phase in 4*['train']+['val']: #['val']:
            bar_st = iterations_per_epoch if phase == 'train' else val_iterations_per_epoch
            bar = pkbar.Pbar(name='update: ', target=bar_st)
            if phase == 'train':
                model.train(True)
                epochs += 1
                torch.autograd.set_grad_enabled(True)
            else:
                model.train(False)  # Set model to evaluate mode
                _ = model.module.aggregate_sub_bn_stats() # FOR EVAL AGGREGATE BN STATS
                torch.autograd.set_grad_enabled(False)

            count = 0

            tot_loss = 0.0
            tot_cls_loss = 0.0
            tot_acc = 0.0
            tot_corr = 0.0
            tot_dat = 0.0
            num_iter = 0
            optimizer.zero_grad()

            # Iterate over data.
            print(phase)
            for i,data in enumerate(trainval_loaders[phase]):
                num_iter += 1
                #bar.update(i)
                if phase == 'train':
                    if i> iterations_per_epoch:
                        break
                    #inputs, labels, long_ind, stats = data
                    inputs, labels, seg_names = data

                    #long_ind = long_ind[0].item()
                    # if long_ind != last_long:
                    #     bn_splits = model.module.update_bn_splits_long_cycle(LONG_CYCLE[long_ind]) # UPDATE BN SPLITS FOR LONG CYCLES
                    #     lr_scale_fact = LONG_CYCLE[long_ind] if (last_long==-2 or long_ind==-1) else LONG_CYCLE_LR_SCALE[long_ind] # WHEN RESTARTING TRAINING AT DIFFERENT LONG CYCLES / AT LAST CYCLE
                    #     last_long = long_ind
                    #     for g in optimizer.param_groups:
                    #         g['lr'] *= lr_scale_fact
                    #         lr = g['lr']
                    #     print_stats(long_ind, batch_size, stats, gamma_tau, bn_splits, lr)
                    # elif RESTART:
                    #     RESTART = False
                    #     print_stats(long_ind, batch_size, stats, gamma_tau, bn_splits, optimizer.state_dict()['param_groups'][0]['lr'])

                else:
                    inputs, labels, seg_names = data
                    # b,n,c,t,h,w = inputs.shape # FOR MULTIPLE TEMPORAL CROPS
                    # inputs = inputs.view(b*n,c,t,h,w)

                inputs = inputs.cuda() # B 3 T W H
                labels = labels.cuda() # B 1

                logits = model(inputs).cpu() # B C 1

                if phase == 'train':
                    #logits_sm = F.softmax(logits, dim=1) # not necessary
                    _, preds = torch.max(logits, 1)
                else:
                    #logits = logits.view(b,n,logits.shape[1],1) # FOR MULTIPLE TEMPORAL CROPS
                    #logits_sm = F.softmax(logits, dim=2)
                    # logits_sm = torch.mean(logits, 1)
                    # logits = torch.mean(logits, 1)
                    #_, preds = torch.max(logits_sm, 1)
                    preds = torch.max(logits, 1)[1]

                #print("- logits : ", logits)

                cls_loss = criterion(logits, labels.cpu())
                tot_cls_loss += cls_loss.item()

                # Calculate top-1 accuracy
                correct = torch.sum(preds.cpu() == labels.data.cpu())
                tot_corr += correct.double()
                tot_dat += logits.shape[0]

                loss = cls_loss/num_steps_per_update
                tot_loss += loss.item()

                if phase == 'train':
                    loss.backward()

                if num_iter == num_steps_per_update and phase == 'train':
                    lr_warmup(lr, steps-st_steps, warmup_steps, optimizer) # USE ONLY AT THE START, AVOID OVERLAP WITH LONG_CYCLE CHANGES
                    steps += 1
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    #lr_sched.step()
                    s_times = iterations_per_epoch//2
                    if (steps-load_steps) % s_times == 0:
                        tot_acc = tot_corr/tot_dat
                        count += 1
                        print (' Epoch:{} {} steps: {} Cls Loss: {:.4f} Tot Loss: {:.4f} Acc: {:.4f}'.format(epochs, phase,
                            steps, tot_cls_loss/(s_times*num_steps_per_update), tot_loss/s_times, tot_acc))
                        tmp_loss = tot_loss/s_times
                        tmp_acc = tot_acc

                        if(count == 2):
                            # print("2. tra : loss", tmp_loss)
                            # print("2. tra : acc", tmp_acc)
                            save_loss.append(tmp_loss)
                            save_acc.append(tmp_acc)
                    
                        tot_loss = tot_cls_loss = tot_acc = tot_corr = tot_dat = 0.
            
                    # if steps % (1000*4) == 0:
                    #     ckpt = {'model_state_dict': model.module.state_dict(),
                    #             'optimizer_state_dict': optimizer.state_dict()
                    #             #'scheduler_state_dict': lr_sched.state_dict(),
                    #             #'long_ind': long_ind}
                    #             }
                    #     torch.save(ckpt, save_model+str(steps).zfill(6)+'.pt')
                    

            if phase == 'val':
                tot_acc = tot_corr/tot_dat
                print (' Epoch:{} {} Cls Loss: {:.4f} Tot Loss: {:.4f} Acc: {:.4f}'.format(epochs, phase,
                    tot_cls_loss/num_iter, (tot_loss*num_steps_per_update)/num_iter, tot_acc))
                # print("1. val, loss", tot_cls_loss/num_iter)
                # print("1. val, acc", tot_acc)
        
                for i in range(4):
                    save_loss_val.append(tot_cls_loss/num_iter)
                    save_acc_val.append(tot_acc)
                    
                valid_acc = tot_acc

                if valid_acc > best_acc:
                    best_acc = valid_acc
                    #torch.save(model.state_dict(), model_path+str(epochs).zfill(6)+'.pt')
                    torch.save(model.state_dict(), model_path+'.pt')
                    print("---------------------------")
                    print('best_acc at Epoch: ', epochs)
                    print('saving model with acc {:.3f}'.format(best_acc))
                    print("---------------------------")

                tot_loss = tot_cls_loss = tot_acc = tot_corr = tot_dat = 0.

        

            if epochs%80 == 0:
                file_name = '/home/dahee333/custom_X3D/model/model_epoch_'+str(epochs).zfill(6)+'.pt'
                torch.save(model.state_dict(), file_name)
                pyplot.plot(save_loss, label='train')
                pyplot.plot(save_loss_val, label='validation')  
                pyplot.legend() 
                pyplot.title('loss')
                pyplot.xlabel('epoch')
                loss_file_name = '/home/dahee333/custom_X3D/loss/loss_epoch_'+str(epochs).zfill(6)+'.png'
                pyplot.savefig(loss_file_name)
                pyplot.clf()
                pyplot.plot(save_acc, label='train')
                pyplot.plot(save_acc_val, label='validation')  
                pyplot.legend() 
                pyplot.title('accuracy')
                pyplot.xlabel('epoch')
                acc_file_name = '/home/dahee333/custom_X3D/accuracy/acc_epoch_'+str(epochs).zfill(6)+'.png'
                pyplot.savefig(acc_file_name)
                pyplot.clf()



def lr_warmup(init_lr, cur_steps, warmup_steps, opt):
    start_after = 1
    if cur_steps < warmup_steps and cur_steps > start_after:
        lr_scale = min(1., float(cur_steps + 1) / warmup_steps)
        for pg in opt.param_groups:
            pg['lr'] = lr_scale * init_lr


if __name__ == '__main__':
    run()
