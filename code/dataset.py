import numpy as np
import imageio

import torch
from PIL import Image
from torch.utils import data
import torchvision.transforms as standard_transforms

ori_data_root = '/projectnb2/ec523/yqz2019/project/environment/ours/datas/'
#aug_data_root = '/home/zyq/Kaggle/datasets/stage1_train_augmented/'

train_num = 90
test_num = 10

def ori_data_path_list(num, mode='train', modality='2D'):
    assert mode in ['train', 'val', 'test']
    items = []
    if modality == '2D':
        if mode == 'train':
            assert num <= 44970
            for i in range(num):
                idx = '{:0>6}'.format(i)
                img_file_path = ori_data_root + '2D/images/' + idx + '.npy'
                mask_file_path = ori_data_root + '2D/labels/' + idx + '.npy'
                items.append((img_file_path, mask_file_path))
        elif mode == 'test':
            assert num <= 454
            for i in range(44970, 44970+num):
                idx = '{:0>6}'.format(i)
                img_file_path = ori_data_root + '2D/images/' + idx + '.npy'
                mask_file_path = ori_data_root + '2D/labels/' + idx + '.npy'
                items.append((img_file_path, mask_file_path))

    elif modality == '3D':
        if mode == 'train':
            assert num <= 44140
            for i in range(train_num):
                idx = '{:0>6}'.format(i)
                img_file_path = ori_data_root + '3D/images/' + idx + '.npy'
                mask_file_path = ori_data_root + '3D/labels/' + idx + '.npy'
                items.append((img_file_path, mask_file_path))
        elif mode == 'test':
            assert num <= 444
            for i in range(44140, 44140+num):
                idx = '{:0>6}'.format(i)
                img_file_path = ori_data_root + '3D/images/' + idx + '.npy'
                mask_file_path = ori_data_root + '3D/labels/' + idx + '.npy'
                items.append((img_file_path, mask_file_path))
    else:
        print('invalid modality')

    return items

class KiTS(data.Dataset):
    def __init__(self, num, mode='train', modality='2D', transform = None, target_transform = None):
        super(KiTS, self).__init__()
        self.num = num
        self.mode = mode
        self.modality = modality
        self.transform = transform
        self.target_transform = target_transform
        self.ori_imgs = ori_data_path_list(num, mode, modality)

    def __getitem__(self, index):
        total_img_num = len(self.ori_imgs)
        if index < total_img_num:
            img_path, mask_path = self.ori_imgs[index]
            img = np.load(img_path)
            img = img.reshape(img.shape[0], img.shape[1], 1)
            mask = np.load(mask_path)
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                mask = self.target_transform(mask).type(torch.LongTensor)
            else:
                mask = torch.from_numpy(mask).type(torch.LongTensor)
        else:
            print('index exceeded!')

        return img, mask

    def __len__(self):
        return len(self.ori_imgs)

def factory(dataset_name, data_num, mode, modality):
    if dataset_name == 'KiTS':
        transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            #standard_transforms.Normalize(*mean_std)
        ])
        dataset = KiTS(data_num, mode=mode, modality=modality,
                transform = transform, target_transform = None)
    else:
        print('wrong dataset name!')
    return dataset

def test_KiTS():
    import torchvision.transforms as standard_transforms
    # train_joint_transform = joint_transforms.Compose([
    #     joint_transforms.Scale_((256, 256)),
    #     #joint_transforms.RandomCrop(256),
    #     #joint_transforms.RandomHorizontallyFlip()
    # ])
    '''
    input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    target_transform = extend_transforms.MaskToTensor()
    '''
    dataset1 = factory('KiTS', 611, mode='train', modality='2D')
    img, mask = dataset1[0]
    print(img.dtype)
    print(mask.dtype)
    print(img.shape, mask.shape)
    # print(len(dataset1))
    # imageio.imwrite('./img.png', img)
    # imageio.imwrite('./mask.png', mask)
    # dataset2 = KiTS(60, 'test', '3D')
    # img, mask = dataset2[9]
    # print(img.shape)
    # print(mask.shape)
    # print(len(dataset2))
    # imageio.imwrite('./timg.png', img[2, :, :])
    # imageio.imwrite('./tmask.png', mask[2, :, :])
    # tmp = mask[2, :, :]
    # print('bg', len(tmp[tmp==0]))
    # print('kd', len(tmp[tmp==1]))
    # print('tm', len(tmp[tmp==2]))

if __name__ == '__main__':
    test_KiTS()
