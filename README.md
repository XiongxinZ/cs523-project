All the codes are in the **code/** folder.

### Dataset Preparation

We use the [KiTS19](https://github.com/neheller/kits19) dataset and preprocessing code provided by [nnUNet](https://github.com/MIC-DKFZ/nnUNet) repository.

1. download the KiTS19 dataset from the above [repository](https://github.com/neheller/kits19). The raw data are in the **data/** folder.

2. Organized the raw data according to the requirements of nnUNet.

We created a script for this. Just run `python3 reorganize_data_folder.py`. nnUNet also requires a json file for data preprocess, run `create_json.py` to generate it. You have to change the path of folders where you want to store the datas accordingly. The lines need to be changed have been marked by comments.

3. Preprocessing

First, you need to install the nnUNet according to the instruction of this [repo](https://github.com/MIC-DKFZ/nnUNet). After doing step 2, run command line `nnUNet_convert_decathlon_task -i FOLDER_TO_TASK_AS_DOWNLOADED_FROM_MSD -p NUM_PROCESSES` and `nnUNet_plan_and_preprocess -t XXX --verify_dataset_integrity` to get the preprocessed data. Detailed descriptions can be found [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_conversion.md).

### Training and testing the model

After the data are ready, run `python3 train.py [task_name]` to train and test the model. [task_name] is the identifier for your training settings. The settings can be changed in train.py file, where marked by comments.
