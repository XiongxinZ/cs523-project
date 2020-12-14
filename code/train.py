import sys
import os
import numpy as np
import imageio
from PIL import Image
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from unet import TwoDUNet
from dataset import factory

# change these begin
DROPOUT_RATIO = 0.1
TRAIN_NUM = 45900
TEST_NUM = 5100
BATCH_SIZE = 4
USE_CUDA = True
EPOCH = 90
# change these end
CHECK_NUM = TRAIN_NUM // (BATCH_SIZE*4)
LUT = np.zeros((3,3),dtype=np.uint8)
LUT[0]=[255,0,0]
LUT[1]=[0,255,0]
LUT[2]=[0,0,255]
device = torch.device("cuda" if USE_CUDA else "cpu")

def metric(predictions, targets, addr=None, idx=None):
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    for i in range(predictions.shape[0]):
        pred = predictions[i].flatten()
        mask = targets[i].flatten()
        pred_kd = np.where(pred==1)
        mask_kd = np.where(mask==1)
        pred_tm = np.where(pred==2)
        mask_tm = np.where(mask==2)
        tp_kd = len(np.intersect1d(pred_kd, mask_kd))
        tp_tm = len(np.intersect1d(pred_tm, mask_tm))
        dice_kd = 2*tp_kd / (len(pred_kd[0]) + len(mask_kd[0]))
        dice_tm = 2*tp_tm / (len(pred_kd[0]) + len(mask_kd[0]))
        dice_cp = (dice_kd + dice_tm) / 2

        if addr != None:
            pred = predictions[i].astype(np.uint8)
            pred= LUT[pred]
            path = addr + str(idx) + '_' + str(i) + '.png'
            imageio.imwrite(path, pred)

    return dice_kd, dice_tm, dice_cp, predictions.shape[0]

class MultiCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=255):
        super(MultiCrossEntropyLoss, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, ignore_index = 255)

    def forward(self, inputs, targets):
        return self.nll_loss(inputs, targets)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def main(exp_name):

    net = TwoDUNet(in_channel=1, nf=8)
    print(net)
    train_set = factory('KiTS', TRAIN_NUM, mode='train', modality='2D')
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                                    num_workers=2, shuffle=True)
    print('dataset:', len(train_loader))
    criterion = MultiCrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    #scheduler = ReduceLROnPlateau(optimizer, 'min', patience=args['lr_patience'], min_lr=1e-10)
    save_dir = './models/' + exp_name + '/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    print('training starts...')
    for epoch in range(EPOCH):
        train(save_dir, train_loader, net, criterion, optimizer, epoch)
    
    net.eval()
    test_set = factory('KiTS', TEST_NUM, mode='test', modality='2D')
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE,
                                    num_workers=2, shuffle=True)
    dice_kd = 0
    dice_tm = 0
    dice_cp = 0
    img_num = 0
    save_dir = './outputs/' + exp_name + '/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for i, data in enumerate(test_loader):
        inputs, masks = data
        inputs = inputs.to(device)
        masks = masks.to(device)
        outputs = net(inputs)
        predictions = outputs.data.max(1)[1].squeeze(1)
        kd_adder, tm_adder, cp_adder, img_num_adder = metric(predictions, masks.data, save_dir, i)
        dice_kd += kd_adder
        dice_tm += tm_adder
        dice_cp += cp_adder
        img_num += img_num_adder
    print('----------------------------------------------------')
    print('test result:  [average dice score %.2f, %.2f, %.2f]' % (
                (dice_kd / img_num), (dice_tm / img_num), (dice_cp / img_num)))

def train(save_dir, train_loader, net, criterion, optimizer, epoch, ):
    
    net.train()
    net.to(device)
    criterion.to(device)
    curr_iter = (epoch-1)*len(train_loader)
    train_loss = AverageMeter()
    dice_kd = 0
    dice_tm = 0
    dice_cp = 0
    img_num = 0
    for i, data in enumerate(train_loader):
        inputs, masks = data
        N = inputs.size(0)
        inputs = inputs.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        predictions = outputs.data.max(1)[1].squeeze(1)
        kd_adder, tm_adder, cp_adder, img_num_adder = metric(predictions, masks.data)
        #print('[iter %d], [score_sum %.5f], [pic_num %d]' % (i, score_adder, img_num_adder))
        dice_kd += kd_adder
        dice_tm += tm_adder
        dice_cp += cp_adder
        img_num += img_num_adder
        loss = criterion(outputs, masks) 
        loss.backward()
        optimizer.step()
        train_loss.update(loss.item(), 1)
        curr_iter += 1
        if (i + 1) % CHECK_NUM == 0:
            print('[epoch %d], [iter %d / %d], [train loss %.5f], [average dice score %.2f, %.2f, %.2f]' % (
                epoch, i + 1, len(train_loader), train_loss.avg,
                (kd_adder / img_num_adder), (tm_adder / img_num_adder), (cp_adder / img_num_adder)))
    
    if epoch%10 == 9:
        torch.save(net.state_dict(), save_dir+str(epoch)+'.pth')
    print('summary:  [epoch %d], [train loss %.5f], [average dice score %.2f, %.2f, %.2f]' % (
                epoch, train_loss.avg, (dice_kd / img_num), (dice_tm / img_num), (dice_cp / img_num)))


def test(num):
    net = TwoDUNet(in_channel=1, nf=16)
    model_dir = os.path.join('experiments', opt.experiments, str(num) + '.pth')
    net.load_state_dict(torch.load(model_dir))

if __name__ == '__main__':
    main(sys.argv[1])
    # for i in range(20, 40):
    #     test(i)
