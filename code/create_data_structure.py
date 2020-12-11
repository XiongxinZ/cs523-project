import os
import numpy as np
import nibabel as nib

root = '/projectnb2/ec523/yqz2019/project/kits19/'  # change this
src = 'data/case_00'

# change this
dst1 = '/projectnb2/ec523/yqz2019/project/environment/ours/datas/2D/'
os.mkdir(dst1)
os.mkdir(dst1 + 'images')
os.mkdir(dst1 + 'labels')
train_num = 210  # change this

cnt = -1
for i in range(0, train_num):
    num = '{:0>3}'.format(i)
    image_src_name = root + src + num + '/imaging.nii.gz'
    label_src_name = root + src + num + '/segmentation.nii.gz'
    imgs = nib.load(image_src_name).get_fdata()
    labels = nib.load(label_src_name).get_fdata()
    for j in range(imgs.shape[0]):
        cnt += 1
        img = imgs[j, :, :]
        label = labels[j, :, :]
        idx = '{:0>6}'.format(cnt)
        image_dst_name = dst1 + 'images/' + idx
        label_dst_name = dst1 + 'labels/' + idx
        np.save(image_dst_name, img)
        np.save(label_dst_name, label)

# change this
dst2 = '/projectnb2/ec523/yqz2019/project/environment/ours/datas/3D/'
os.mkdir(dst2)
os.mkdir(dst2 + 'images')
os.mkdir(dst2 + 'labels')
train_num = 210  # change this
neighbor = 5  # change this

cnt = -1
for i in range(0, train_num):
    num = '{:0>3}'.format(i)
    image_src_name = root + src + num + '/imaging.nii.gz'
    label_src_name = root + src + num + '/segmentation.nii.gz'
    imgs = nib.load(image_src_name).get_fdata()
    labels = nib.load(label_src_name).get_fdata()
    for j in range(imgs.shape[0] - neighbor + 1):
        cnt += 1
        img = imgs[j:j+neighbor, :, :]
        label = labels[j:j+neighbor, :, :]
        idx = '{:0>6}'.format(cnt)
        image_dst_name = dst2 + 'images/' + idx
        label_dst_name = dst2 + 'labels/' + idx
        np.save(image_dst_name, img)
        np.save(label_dst_name, label)