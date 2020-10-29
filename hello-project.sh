#!/bin/bash

# installation the executable file on scc user
pip3 install --user nnunet

git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet

pip3 install --user -e .

# setup folders for data processing on scc
cd ..
mkdir -p ./dataset
mkdir 