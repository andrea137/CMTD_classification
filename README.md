# CMTD_classification

Original implementation of the code of the paper "Digital and computer-aided pathology classification of canine mammary tumors". To be published...


## Create the [conda](https://www.anaconda.com/) enviroment

- Linux:

    conda env create -f environment.yml

- Windows:

    conda env create -f environment_win.yml

## Prepare the dataset using the code in "prepare_folds", or manually using the common train val folders approach (e.g., like [ImageNet](http://image-net.org/)) and change the variable data_dir accordingly 

## TO run the code 

python -u main.py vethis -a vgg16 2>&1 | tee out.log

## For more options: 

python main.py -h
