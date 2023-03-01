#!/usr/bin/env python
# coding: utf-8


import os 
from pathlib import Path
import multiprocessing
import os 
import numpy as np
import pandas as pd
import multiprocessing
from collections import Counter
from matplotlib import pyplot as plt
from shutil import rmtree
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

create_folds = False

# replace username with yours
data_dir = Path('c:/Users/username/work/vet/images_def')
#data_dir = Path('/home/username/datasets/vet/images_def')
num_workers = multiprocessing.cpu_count() - 2

# Importing all necessary libraries 
print(os.listdir(data_dir))

filenames = data_dir.glob('*/*.jpg')
df = pd.DataFrame({'filename' : list(filenames)})
df['class'] = df['filename'].apply(lambda x: x.parents[0].stem) 
df['patient'] = df['filename'].apply(lambda x: x.stem.split('_FRM_')[0])

#df = df.loc[df['class'] != 'sana',]

print(df['class'].value_counts())
print(len(df['patient'].value_counts()))
print(df['patient'].value_counts())

folds = [f'fold{i}' for i in range(1,6)]
print(folds)

if create_folds:
    group_kfold = GroupKFold(n_splits=len(folds))
    group_kfold.get_n_splits(df, df['class'], df['patient'])
    for i, (train_index, test_index) in enumerate(group_kfold.split(df, df['class'], df['patient'])):
        print("TRAIN:", train_index, "TEST:", test_index)
        print("TRAIN:", type(train_index), "TEST:", type(test_index))
        fold = f'fold{i+1}'
        df[fold] = False
        df.iloc[test_index, df.columns.get_loc(fold)] = True 
        assert set(df.loc[df[fold], 'patient']).isdisjoint(df.loc[~df[fold], 'patient'])
    print(df)
    df.to_csv('folds.csv')

else:
    df2 = pd.read_csv('folds.csv')
    df2['filename'] = df2['filename'].apply(lambda x: Path(x))
    df2['chkname'] = df2['filename'].apply(lambda x: x.name)
    df['chkname'] = df['filename'].apply(lambda x: x.name)
    df = df.sort_values(by=['chkname']).reset_index(drop=True)
    df2_sorted = df2.copy().sort_values(by=['chkname']).reset_index(drop=True)
    print(df2_sorted.head())
    print(df.head())
    assert df[['chkname', 'class', 'patient']].equals(df2_sorted[['chkname', 'class', 'patient']])
    
    df = pd.merge(df, df2, on='chkname')

    print(df.head())
    print(df.columns)
    assert df['class_x'].equals(df['class_y'])
    assert df['patient_x'].equals(df['patient_y'])
    df = df.drop(['Unnamed: 0', 'filename_y', 'class_y', 'patient_y'], axis='columns')
    df.rename(lambda x: x.replace('_x', ''), axis='columns', inplace=True)
    print(df.columns)

def create_symlink(fnames, out_dir):
    for fname in fnames:
        n_out_dir = out_dir/fname.parents[0].stem
        n_out_dir.mkdir(exist_ok=True)
        dest_fname = n_out_dir/fname.name
        os.symlink(fname, dest_fname)
    

for fold in folds:
    train_dir = Path.cwd()/fold/'train'
    train_dir.mkdir(parents=True)
    val_dir = Path.cwd()/fold/'val'
    val_dir.mkdir(parents=True)
    print(train_dir, val_dir)
    train_fnames = list(df.loc[~df[fold], 'filename'])
    val_fnames = list(df.loc[df[fold], 'filename'])
    #print(train_fnames[0].parents[0].stem)
    #print(val_fnames[0])
    assert set(train_fnames).isdisjoint(val_fnames)
    create_symlink(train_fnames, train_dir)
    create_symlink(val_fnames, val_dir)
    



