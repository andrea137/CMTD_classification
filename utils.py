import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm 
import pickle
import statsmodels.stats.api as sms
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
import pandas as pd
from pathlib import Path

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)

def save_pickle(save_dir, dct, save_name):
    save_dir.mkdir(exist_ok=True)
    with open(save_dir/save_name, "wb") as f:
        pickle.dump(dct,f)

def read_pickle(read_dir, read_name):
    with open(read_dir/read_name,"rb") as f:
        loaded = pickle.load(f)
    return loaded

def read_saved_data(read_dir, phase):
    X = read_pickle(read_dir, f'X_{phase}.pkl')
    y = read_pickle(read_dir, f'y_{phase}.pkl')
    paths = read_pickle(read_dir, f'{phase}_paths.pkl')
    return X, y, paths

def save_data(save_dir, X, y, paths, phase):
    save_pickle(save_dir, X, f'X_{phase}.pkl')
    save_pickle(save_dir, y, f'y_{phase}.pkl')
    save_pickle(save_dir, paths, f'{phase}_paths.pkl')
    

def cross_validate(est, folds, features_dir, data, p_dist, n_iter_search, dataset_name, workers=-1):
    results = {'score_list' : [], 'cm_list' : [], 'recognition_rate' : [],
               'tc_list' : [], 'tc_cm_list' : [], 'tc_recognition_rate' : [],
               'cv_history' : [], 'best_params' : []}

    for fold in tqdm(folds):
        X_train, y_train, train_paths = read_saved_data(features_dir/fold, 'train')
        X_val, y_val, val_paths = read_saved_data(features_dir/fold, 'val')
        X_test, y_test, test_paths = read_saved_data(features_dir/fold, 'test')
        
        # inner cross validation to find best parameters over the training set
        inner_cv = GroupKFold(n_splits=5)
        clf = RandomizedSearchCV(estimator=est, param_distributions=p_dist, cv=inner_cv, n_iter=n_iter_search, n_jobs=workers, random_state=42)
        train_groups = get_patient_from_filename(train_paths, dataset_name)
        clf.fit(X_train, y_train, groups=train_groups)

        results['score_list'].append(clf.score(X_val, y_val))
        y_pred = clf.predict(X_val)
        results['cm_list'].append(confusion_matrix(y_val, y_pred))# TODO:, labels=))
        results['recognition_rate'].append(get_recognition_rate(data[fold], dataset_name,
                                           y_pred, y_val, 'val'))
        tc, tc_cm, tc_preds = compute_ten_crops_accuracy(clf, X_test, y_test, with_preds=True)
        results['tc_list'].append(tc)
        results['tc_cm_list'].append(tc_cm)
        results['tc_recognition_rate'].append(get_recognition_rate(data[fold], dataset_name, tc_preds, y_test, 'test'))
        results['cv_history'].append(clf.cv_results_)
        results['best_params'].append(clf.best_params_)
    return results

def get_ten_crops_preds(clf, X_test, y_test, debug=False):
    preds = clf.predict(X_test)
    sum_preds = np.reshape(preds, (-1, 10)).sum(1)
        
    new_preds = [1 if p >= 5 else 0 for p in sum_preds]
    
    if debug:
        for i in range(len(y_test)):
            print(np.reshape(preds, (-1, 10))[i], sum_preds[i], new_preds[i], y_test[i])

    return new_preds

def compute_ten_crops_accuracy(clf, X_test, y_test, with_preds=False, debug=False):
    
    new_preds = get_ten_crops_preds(clf, X_test, y_test, debug=debug)
    
    tc = accuracy_score(y_test, new_preds)
    cm = confusion_matrix(y_test, new_preds)
    if with_preds:
        return tc, cm, new_preds
    return tc, cm

def get_accuracy(cm):
    tn, fp, fn, tp = cm.ravel()
    return (tp+tn)/(tn + fp + fn + tp)

def get_ppv(cm):
    tn, fp, fn, tp = cm.ravel()
    return tp/(tp + fp)

def get_npv(cm):
    tn, fp, fn, tp = cm.ravel()
    return tn/(tn + fn) 

def get_sensitivity(cm):
    tn, fp, fn, tp = cm.ravel()
    return tp/(tp + fn)

def get_specificity(cm):
    tn, fp, fn, tp = cm.ravel()
    return tn/(tn + fp) 

def print_average_ci(lst, decimals=2, title=''):
    print(title, np.around(np.mean(lst), decimals=decimals), 
          '+-', np.around(np.std(lst), decimals=decimals), 
          'ci', np.around(sms.DescrStatsW(lst).tconfint_mean(), decimals=decimals))


def store_best(cur_res, cur_best, cv_results, params):
    """If improved store new best"""
    best = cur_best
    if  cur_res > cur_best['score']:
        best['score'] = cur_res
        best['score_list'] = cv_results['score_list']
        best['cm_list'] = cv_results['cm_list']
        best['recognition_rate'] = cv_results['recognition_rate']
        best['tc_list'] = cv_results['tc_list']
        best['tc_cm_list'] = cv_results['tc_cm_list']
        best['tc_recognition_rate'] = cv_results['tc_recognition_rate']
        best['params'] = params
    return best


def compute_averages_ci(cms):
    """cms: list of confusion matrices"""
    acc = list(map(get_accuracy, cms))
    ppv = list(map(get_ppv, cms))
    npv = list(map(get_npv, cms))
    tpr = list(map(get_sensitivity, cms))
    tnr = list(map(get_specificity, cms))
    #print(acc, ppv, npv)
    print_average_ci(acc, title='ACC')
    print_average_ci(ppv, title='PPV')
    print_average_ci(npv, title='NPV')
    print_average_ci(tpr, title='TPR')
    print_average_ci(tnr, title='TNR')
    #print('Acc', np.mean(acc), '+-', np.std(acc), 'ci', sms.DescrStatsW(acc).tconfint_mean())
    #print('PPV', np.mean(ppv), '+-', np.std(ppv), 'ci', sms.DescrStatsW(ppv).tconfint_mean())
    #print('NPV', np.mean(npv), '+-', np.std(npv), 'ci', sms.DescrStatsW(npv).tconfint_mean())
    #print('TPR', np.mean(tpr), '+-', np.std(tpr), 'ci', sms.DescrStatsW(tpr).tconfint_mean())
    #print('TNR', np.mean(tnr), '+-', np.std(tnr), 'ci', sms.DescrStatsW(tnr).tconfint_mean())

def get_patient_from_filename(filenames, dataset_name, debug=False):
    if dataset_name == 'breakhis':
        split_at = '-'
        end_idx = 3
    elif dataset_name == 'vethis':
        split_at = '_FRM'
        end_idx = 1
    else:
        raise NotImplementedError()
    df = pd.DataFrame(filenames, columns=['filename'])
    df['patient'] = df['filename'].apply(lambda x: ''.join(Path(x).stem.split(split_at)[0:end_idx]))
    if debug:
        print(len(df['patient'].value_counts()))
        print(df['patient'].value_counts())
    return list(df['patient'])

def get_recognition_rate(data, dataset_name, y_preds, y_chk, phase='val'):
    if dataset_name == 'breakhis':
        split_at = '-'
        end_idx = 3
    elif dataset_name == 'vethis':
        split_at = '_FRM'
        end_idx = 1
    else:
        raise NotImplementedError()

    #print(data.image_datasets[phase].classes)

    df = pd.DataFrame(data.image_datasets[phase].imgs, 
        columns=['filename', 'y_true'])
    df['patient'] = df['filename'].apply(lambda x: ''.join(Path(x).stem.split(split_at)[0:end_idx]))
    #df['patient'] = df['filename'].apply(lambda x: ''.join(Path(x).stem.split('_FRM')[0]))
    df['y_preds'] = y_preds
    #print(df.head())
    #print(df['filename'][0])
    # verify that the order is correct (it should be because shuffle is set to False in teh dataloaders)
    np.testing.assert_array_equal(df['y_true'], y_chk)

    # Drop unnecessary column and create a list of values per patient
    by_patient = df.drop(['filename'], axis=1).groupby('patient').agg(list).reset_index()
    # Compute patient score
    by_patient['score'] = by_patient.apply(lambda x: accuracy_score(x['y_true'], x['y_preds']), axis=1)

    #print(''.join([str(row) for row in by_patient.iterrows()]))

    return by_patient['score'].mean()

def print_best(best):
    print(best['score_list'])
    print(best['cm_list'])
    print(best['recognition_rate'])
    print(best['tc_list'])
    print(best['tc_cm_list'])
    print(best['tc_recognition_rate'])
    try:
        print("Best_params")
        print(best['best_params'])
    except:
        pass
    
    print("Best score")
    print_average_ci(best['score_list'])
    compute_averages_ci(best['cm_list'])
    print("Recognition rate")
    print_average_ci(best['recognition_rate'])
    print("Best ten crop")
    print_average_ci(best['tc_list'])
    compute_averages_ci(best['tc_cm_list'])
    print("Ten crops recognition rate")
    print_average_ci(best['tc_recognition_rate'])
    

def print_best_from_file(read_dir='results', read_name='history.pkl', metrics='score_list'):
    assert metrics in ['score_list', 'tc_list', 'recognition_rate', 'tc_recognition_rate']
    read_dir=Path(read_dir)
    results = read_pickle(read_dir, read_name)
    best = max(results, key=lambda x:np.mean(x[0][metrics])) 
    print(best)
    print_best(best[0])
    print(f"Params: {best[1]}")
