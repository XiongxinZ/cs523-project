import json

dct = { 
 "name": "KiTS", 
 "description": "kidney and kidney tumor segmentation",
"reference": "KiTS data for nnunet",
"licence":"",
"relase":"0.0",
"tensorImageSize": "3D",
"modality": { 
   "0": "CT"
 }, 
 "labels": { 
   "0": "background", 
   "1": "Kidney",
   "2": "Tumor"
 }, 
"training":[],
 "test": []}

root = "./"

train_num = 7  # change this
test_num = 3  # change this

dct["numTraining"] = train_num
dct["numTest"] = test_num

for i in range(0, train_num):
    item = dict()
    item["image"] = root + 'imagesTr/case_' + '{:0>3}'.format(i) + '_0000.nii.gz'
    item["label"] = root + 'labelsTr/case_' + '{:0>3}'.format(i) + '_0000.nii.gz'
    dct["training"].append(item)

for i in range(train_num, train_num+test_num):
    dct["test"].append(root + 'imagesTs/case_' + '{:0>3}'.format(i) + '_0000.nii.gz')

fptr = open('./nnUNet_raw_data_base/nnUNet_raw_data/Task42_KiTS/dataset.json', 'w')  # change this
json.dump(dct, fptr)
