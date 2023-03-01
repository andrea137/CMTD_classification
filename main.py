#!/usr/bin/env python
# Usage:  python -u main.py vethis -a vgg16 2>&1 | tee out.log

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import ParameterGrid, GroupKFold, GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform, loguniform
import numpy as np
from pathlib import Path
from multiprocessing import cpu_count
from data import Data
from utils import *
from hooks import *
import torch
from torchvision import models
from torchvision.utils import make_grid
from efficientnet_pytorch import EfficientNet

from tqdm import tqdm 
import time
import argparse

def parse():
    model_names = ['vgg16',
                   'vgg16bn',
                   'vgg19',
                   'inception',
                   'effnet',
    ]
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', metavar='DS',
                        help='which dataset to use, breakhis or vethis')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg16',
                        choices=model_names,
                        help='cnn architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=None, type=int, metavar='N',
                        help='number of threads to use (default: None)')
    parser.add_argument('-rs', '--resize_size', default=512, type=int, metavar='RS',
                        help='images are resized to this value (default: 512)')
    parser.add_argument('-cs', '--crop_size', default=224, type=int, metavar='CS',
                        help='images are cropped to this value after resize (default: 224, 299 for inception)')
    parser.add_argument('-nl', '--n_loops', default=1, type=int, metavar='NL',
                        help='number of loops of data augmentation over the training set (default: 1)')
    parser.add_argument('-tr', '--trns', default='base', metavar='TR',
                        help='transforms for data augmentation (default: base, alternative: imagenet)')
    parser.add_argument("--read_features", action="store_true" , help="read features from file (default: generate new features")
    parser.add_argument("--no-linearsvc", action="store_true" , help="Use LinearSVC (default: True)")
    parser.add_argument("--polysvc", action="store_true" , help="Use  Poly SVC (default: False)")
    parser.add_argument("--rbfsvc", action="store_true" , help="Use  RBF SVC (default: False)")
    parser.add_argument("--no-nypolysvc", action="store_true", help="Use  Poly SVC with Nystroem approximation (default: True)")
    parser.add_argument("--no-nyrbfsvc", action="store_true" , help="Use  RBF SVC with Nystroem approximation (default: True)")
    parser.add_argument("--no-xgboost", action="store_true" , help="Use  xgboost (default: True)")
    args = parser.parse_args()
    return args


def main():
    # Define inputs
    args = parse()
    print(args)

    #data_dir = Path('/mnt/lavoro/work/datasets/')
    data_dir = Path('c:/Users/andre/work/')
    if args.dataset == 'breakhis':
        dataset_name=args.dataset
        image_dir = data_dir/'mkfold_imagenet'
        magnification='200X'
    elif args.dataset == 'vethis':
        dataset_name=args.dataset        
        image_dir = data_dir/'vet/images_imagenet_def'
        magnification=''
    else:
        raise NotImplementedError()

    working_dir = Path.cwd()
    features_dir = working_dir #/model_name
    

    model_name = args.arch
    num_workers = 0 #cpu_count() - 2 for data reading
    nfolds = 5
    batch_size = 1 # Not important since we are using the net as a feature extractor
    resize_size = args.resize_size
    crop_size = args.crop_size
    if model_name == 'inception':
        #resize_size = 320
        if crop_size < 299:
            crop_size = 299
    n_loops = args.n_loops
    trns = args.trns 
    generate_features = not args.read_features
    exp_n_feat = 1472 # vgg


    print(f"{dataset_name}_{model_name}_{resize_size}_{crop_size}_aug_{trns}_{n_loops}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if model_name == 'vgg16':
        model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=True)
        # print(model._modules)
        # vgg16
        # Indexes of convolutions
        # layers_idxs = [2, 7, 14, 21, 28]
        # indexes of relu just +1
        layers_name_list = ['features']
        layers_idxs = [3, 8, 15, 22, 29]
        print([model._modules[layers_name_list[0]][i] for i in layers_idxs])
    elif model_name == 'vgg16bn':
        model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16_bn', pretrained=True)
        print(model._modules)
        layers_name_list = ['features']
        layers_idxs = [5, 12, 22, 32, 42]
        print([model._modules[layers_name_list[0]][i] for i in layers_idxs])
    elif model_name == 'vgg19':
        model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19', pretrained=True)
        print(model._modules)
        layers_name_list = ['features']
        layers_idxs = [3, 8, 17, 26, 35]
        print([model._modules[layers_name_list[0]][i] for i in layers_idxs])
    elif model_name == 'inception':
        model = torch.hub.load('pytorch/vision:v0.6.0', 'inception_v3', pretrained=True)
        layers_name_list = ['Conv2d_2b_3x3', 'Conv2d_4a_3x3', 'Mixed_5d', 'Mixed_6e'] #, 'Mixed_7c']
        layers_idxs = None
        exp_n_feat = 3360 - 2048

    elif model_name == 'effnet':
        model = EfficientNet.from_pretrained('efficientnet-b4') 
        layers_name_list = ['_blocks'] #, '_bn1'] #'_conv_head']  better with _bn1
        
        # b0 #################################################
        #layers_idxs = [1, 3, 5, 8, 11, 15] # after expansion
        #layers_idxs = [0, 2, 4, 7, 10, 14] # before expansion
        ######################################################
        # b4
        layers_idxs = [1, 5, 9, 15, 21, 29, 31]
        ch_sizes = [24, 32, 56, 112, 160, 272, 448] #, 1792] # 1280 is for the head
        exp_n_feat = np.sum(ch_sizes)
        print([model._modules[layers_name_list[0]][i] for i in layers_idxs])
        #print(model._modules[layers_name_list[1]])
        #raise Exception()
    else:
        raise NotImplementedError()

    model.name = model_name

    # Create folds of features
    folds = [f'fold{i}' for i in range(1,nfolds+1)]
    data = { fold : Data(image_dir, magnification, fold, batch_size, 
            resize_size, crop_size, num_workers) for fold in folds}
    if generate_features:
        for fold in folds:
            X_train, y_train, train_paths = extract_features(model, device, data[fold], 'train', layers_name_list, layers_idxs, n_loops=n_loops, exp_n_feat=exp_n_feat)
            X_val, y_val, val_paths = extract_features(model, device, data[fold], 'val', layers_name_list, layers_idxs, exp_n_feat=exp_n_feat)
            X_test, y_test, test_paths = extract_features(model, device, data[fold], 'test', layers_name_list, layers_idxs, exp_n_feat=exp_n_feat)

            save_data(features_dir/fold, X_train, y_train, train_paths, 'train')
            save_data(features_dir/fold, X_val, y_val, val_paths, 'val')
            save_data(features_dir/fold, X_test, y_test, test_paths, 'test')
            
    del model


    C_range = loguniform(1e-6, 1e1)
    if not args.no_linearsvc:
        print("Linear Support Vector Machine")
        p_dist = dict(linearsvc__C=C_range)
        n_iter_search = 20
        est = make_pipeline(StandardScaler(), LinearSVC(max_iter=10000, class_weight='balanced', random_state=42))
        #print(est)
        cv_results = cross_validate(est, folds, features_dir, data, p_dist, n_iter_search, dataset_name, workers=args.workers)
        save_pickle(working_dir/'results', cv_results, 'svm_linear_history.pkl')
        print_best(cv_results)
        print(cv_results['cv_history'])

    gamma_range = loguniform(1e-5, 1e-2)

    n_iter_search = 40
    p_dist = dict(svc__C=C_range, svc__gamma=gamma_range)

    if args.polysvc:

        print("Polynomial Support Vector Machine")

        est = make_pipeline(StandardScaler(), SVC(kernel='poly', class_weight='balanced'))
        #print(est)
        cv_results = cross_validate(est, folds, features_dir, data, p_dist, n_iter_search, dataset_name, workers=args.workers)
        print_best(cv_results)

        save_pickle(working_dir/'results', cv_results, 'svm_poly_history.pkl')

    if args.rbfsvc:
        print("RBF Support Vector Machine")

        est = make_pipeline(StandardScaler(), SVC(kernel='rbf', class_weight='balanced'))
        #print(est)
        cv_results = cross_validate(est, folds, features_dir, data, p_dist, n_iter_search, dataset_name, workers=args.workers)
        print_best(cv_results)

        save_pickle(working_dir/'results', cv_results, 'svm_rbf_history.pkl')

    
    n_components = 2000
    if n_loops == 1:
        n_components = 500
    
    p_dist = dict(linearsvc__C=C_range, nystroem__gamma=gamma_range)

    if not args.no_nypolysvc:
        print('Linear Support Vector Machine with Nystroem kernel poly approximation')

        est = make_pipeline(StandardScaler(),
                            Nystroem(kernel='poly', degree=3, 
                                    random_state=1, n_components=n_components), 
                            LinearSVC(max_iter=10000, class_weight='balanced', random_state=42)
                            )
        #print(est)
        cv_results = cross_validate(est, folds, features_dir, data, p_dist, n_iter_search, dataset_name, workers=args.workers)
        print_best(cv_results)
        save_pickle(working_dir/'results', cv_results, 'svm_poly_nys_history.pkl')

    if not args.no_nypolysvc:
        print('Linear Support Vector Machine with Nystroem kernel rbf approximation')

        est = make_pipeline(StandardScaler(),
                            Nystroem(random_state=1, n_components=n_components), 
                            LinearSVC(max_iter=10000, class_weight='balanced', random_state=42)
                            )
        #print(est)
        cv_results = cross_validate(est, folds, features_dir, data, p_dist, n_iter_search, dataset_name, workers=args.workers)
        print_best(cv_results)
        save_pickle(working_dir/'results', cv_results, 'svm_rbf_nys_history.pkl')

    if not args.no_xgboost:
        print("XGB")
        n_iter_search = 100
        p_dist = {
                'xgbclassifier__min_child_weight': range(1, 11),
                'xgbclassifier__gamma': uniform(0.5, 5),
                'xgbclassifier__subsample': np.linspace(0.6, 1.0, 5),
                'xgbclassifier__colsample_bytree': np.linspace(0.6, 1.0, 5),
                'xgbclassifier__max_depth': range(1, 10)
                }

        est = make_pipeline(StandardScaler(), 
                            XGBClassifier(objective='binary:logistic',
                            #tree_method='gpu_hist',
                            ))
        cv_results = cross_validate(est, folds, features_dir, data, p_dist, n_iter_search, dataset_name, workers=args.workers)
        print_best(cv_results)    
        save_pickle(working_dir/'results', cv_results, 'xgboost_history.pkl')


if __name__ == '__main__':
    main()
