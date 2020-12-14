import os
from shutil import copyfile

root = '/projectnb2/ec523/yqz2019/project/kits19/'
src = 'data/case_00'
# change this
dst = '/projectnb2/ec523/yqz2019/project/environment/nnUNet/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task41_KiTS/'

os.mkdir(dst)
os.mkdir(dst + 'imagesTr')
os.mkdir(dst + 'imagesTs')
os.mkdir(dst + 'labelsTr')
train_num = 70  # change this
test_num = 30  # change this

for i in range(0, train_num):
  idx = '{:0>3}'.format(i)
  image_src_name = root + src + idx + '/imaging.nii.gz'
  image_dst_name = dst + 'imagesTr/case_' + idx + '_0000.nii.gz'
  copyfile(image_src_name, image_dst_name)
  label_src_name = root + src + idx + '/segmentation.nii.gz'
  label_dst_name = dst + 'labelsTr/case_' + idx + '_0000.nii.gz'
  copyfile(label_src_name, label_dst_name)

for i in range(train_num, train_num+test_num):
    idx = '{:0>3}'.format(i)
    image_src_name = root + src + idx + '/imaging.nii.gz'
    image_dst_name = dst + 'imagesTs/case_' + idx + '_0000.nii.gz'
    copyfile(image_src_name, image_dst_name)