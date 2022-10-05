import torch
import argparse
from hoc import *
import random
import argparse
import numpy as np
from COMPAS.utils import check_T
import pandas as pd


# Options ----------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 1)')
parser.add_argument('--G', type=int, default=50, help='num of rounds (parameter G in Algorithm 1)')
parser.add_argument('--max_iter', type=int, default=1500, help='num of iterations to get a T')
parser.add_argument("--local", default=False, action='store_true')
parser.add_argument("--data_path", default='./COMPAS/results/compas_ce_compas_score.pt')
parser.add_argument("--clip_vec", default=False, action='store_true')
parser.add_argument("--soft", default=False, action='store_true')



def dp_violation(y_pred, attribute, T = None, p = None, CLIP_FLAG = True):
# exact_dp https://arxiv.org/pdf/2109.13642.pdf
    result_mat = np.zeros((len(np.unique(y_pred)), len(np.unique(attribute))))
    for i in np.unique(attribute):
        loc = attribute == i
        for k in np.unique(y_pred):   
            if len(loc.shape) > 1:
                result_mat[i][k] = np.sum((y_pred == k).reshape(-1,1) * loc) / np.sum(loc)
            else:
                result_mat[i][k] = np.sum((y_pred == k) * loc) / np.sum(loc)
    if T is not None:
        if isinstance(T, list): # local
            cnt = 0
            for k in np.unique(y_pred):  
                noisy_p = np.array([np.mean((attribute==i)*1.0) for i in np.unique(attribute)])
                diag_p_inv = np.linalg.inv(np.diag(p.reshape(-1)))
                T_trans_inv = np.linalg.inv(np.transpose(T[cnt]))
                noisy_f = result_mat[:,k].reshape(-1,1)
                # print(np.dot(diag_p_inv, T_trans_inv))
                correct_f = np.dot( np.dot(diag_p_inv, T_trans_inv),  noisy_f * noisy_p.reshape(-1,1))
                print(correct_f) 
                result_mat[:,k] = correct_f.reshape(-1)
                cnt += 1
        else: # global


            for k in np.unique(y_pred):  
                noisy_p = np.array([np.mean((attribute==i)*1.0) for i in np.unique(attribute)])
                diag_p_inv = np.linalg.inv(np.diag(p.reshape(-1)))
                T_trans_inv = np.linalg.inv(np.transpose(T))
                noisy_f = result_mat[:,k].reshape(-1,1)
                # print(np.dot(diag_p_inv, T_trans_inv))
                correct_f = np.dot( np.dot(diag_p_inv, T_trans_inv),  noisy_f * noisy_p.reshape(-1,1))
                print(correct_f) 
                result_mat[:,k] = correct_f.reshape(-1)


    if CLIP_FLAG:
        result_mat = np.clip(result_mat, 1e-3, 1)
        result_mat = result_mat / np.sum(result_mat,1).reshape(-1,1)
        print(f'After Clipping: {result_mat}')
    else:
        print(f'Without Clipping: {result_mat}')

    dp = 0
    for i in range(result_mat.shape[0]):
        for j in range(i+1,result_mat.shape[0]):
            dp += np.sum(np.abs(result_mat[i] - result_mat[j]))
    return dp
        

def eo_violation(y_pred, y_true, attribute, T = None, p = None, CLIP_FLAG = True):
# https://proceedings.neurips.cc/paper/2016/file/9d2682367c3935defcb1f9e247a97c0d-Paper.pdf
    result_mat = np.zeros((len(np.unique(attribute)), len(np.unique(y_pred))*len(np.unique(y_true))))
    for i in np.unique(attribute):
        cnt = 0
        for y in np.unique(y_true):  
            for k in np.unique(y_pred):
                if len(attribute.shape) > 1:
                    loc = (attribute == i) * (y_true == y).reshape(-1,1)
                    result_mat[i][cnt] = np.sum((y_pred == k).reshape(-1,1) * loc) / np.sum(loc)
                else:
                    loc = (attribute == i) * (y_true == y)
                    result_mat[i][cnt] = np.sum((y_pred == k) * loc) / np.sum(loc)
                cnt += 1
    if T is not None:
        if isinstance(T, list): # local
            cnt = 0
            for y in np.unique(y_true):  
                for k in np.unique(y_pred):
                    noisy_p = np.array([np.sum((attribute==i)*(y_true==y).reshape(-1,1))/np.sum(y_true==y)/attribute.shape[1] for i in np.unique(attribute)])
                    diag_p_inv = np.linalg.inv(np.diag(np.array(p[cnt]).reshape(-1)))
                    T_trans_inv = np.linalg.inv(np.transpose(T[cnt]))
                    noisy_f = result_mat[:,cnt].reshape(-1,1)
                    correct_f = np.dot( np.dot(diag_p_inv, T_trans_inv),  noisy_f * noisy_p.reshape(-1,1)) 
                    result_mat[:,cnt] = correct_f.reshape(-1)
                    cnt += 1

        elif isinstance(p, list): # global
            cnt = 0
            for y in np.unique(y_true):  
                for k in np.unique(y_pred):
                    noisy_p = np.array([np.sum((attribute==i)*(y_true==y).reshape(-1,1))/np.sum(y_true==y)/attribute.shape[1] for i in np.unique(attribute)])
                    diag_p_inv = np.linalg.inv(np.diag(np.array(p[cnt]).reshape(-1)))
                    T_trans_inv = np.linalg.inv(np.transpose(T))
                    noisy_f = result_mat[:,cnt].reshape(-1,1)
                    correct_f = np.dot( np.dot(diag_p_inv, T_trans_inv),  noisy_f * noisy_p.reshape(-1,1)) 
                    result_mat[:,cnt] = correct_f.reshape(-1)
                    cnt += 1


    if CLIP_FLAG:
        result_mat = np.clip(result_mat, 1e-3, 1)
        len_pred = len(np.unique(y_pred))
        for y in np.unique(y_true):  
            tmp = result_mat[:,y*len_pred:(y+1)*len_pred]
            tmp = tmp / np.sum(tmp,1).reshape(-1,1)
            result_mat[:,y*len_pred:(y+1)*len_pred] = tmp
        print(f'After Clipping: {result_mat}')
    else:
        print(f'Without Clipping: {result_mat}')

    # eo = np.max(result_mat,0) - np.min(result_mat,0)
    eo = 0.0

    for i in range(result_mat.shape[0]):
        for j in range(i+1,result_mat.shape[0]):
            eo += np.sum(np.abs(result_mat[i] - result_mat[j]))

    if result_mat.shape[0] == 2:
        eop = 0.0
        result_mat = result_mat[:,-1]
        for i in range(result_mat.shape[0]):
            for j in range(i+1,result_mat.shape[0]):
                eop += np.sum(np.abs(result_mat[i] - result_mat[j]))
        return np.array([eo, eop])
    else:
        return eo

            
        
def eval_fairness(config, method, data_eval, metric='eo'):
    print(f'-------------\ncurrent label classifier is {method}')
    # equal opportunity https://proceedings.neurips.cc/paper/2016/file/9d2682367c3935defcb1f9e247a97c0d-Paper.pdf
    # equal odds https://proceedings.neurips.cc/paper/2016/file/9d2682367c3935defcb1f9e247a97c0d-Paper.pdf

    y_pred = data_eval['y_pred']
    y_true = data_eval['y_true']
    noisy_attribute = data_eval['noisy_attribute']
    true_attribute = data_eval['true_attribute']


    if metric == 'dp':
        # baseline: direct estimate with noisy attributes
        noisy_dp = dp_violation(y_pred=y_pred, attribute=noisy_attribute[:,0].reshape(-1), CLIP_FLAG = config.clip_vec)   # only use the first noisy label
        # Noisy attributes + Correction with HOC (assuming conditional independence)
        correct_dp_est = dp_violation(y_pred=y_pred, attribute=noisy_attribute, T = data_eval['T_est'], p = data_eval['p_est'], CLIP_FLAG = config.clip_vec)
        # Noisy attributes + Correction with true T,p (assuming conditional independence)
        correct_dp_true = dp_violation(y_pred=y_pred, attribute=noisy_attribute, T = data_eval['T_true'], p = data_eval['p_true'], CLIP_FLAG = config.clip_vec)
        # Ground-truth: evaluate fairness with clean attributes
        true_dp = dp_violation(y_pred=y_pred, attribute=data_eval['true_attribute'], CLIP_FLAG = config.clip_vec)

        error_base = noisy_dp
        error_est = correct_dp_est
        error_true = correct_dp_true

        # without conditional independence
        T_est, p_est, T_true, p_true = [], [], [], []
        for k in np.unique(y_pred):
            loc = y_pred == k
            T_est_tmp, p_est_tmp, T_true_tmp, p_true_tmp = get_T_p(config, noisy_attribute[loc], lr = 0.1, true_attribute = true_attribute[loc])
            T_est.append(T_est_tmp)
            p_est.append(p_est_tmp)
            T_true.append(T_true_tmp)
            p_true.append(p_true_tmp)
        
        correct_dp_est_fine = dp_violation(y_pred=y_pred, attribute=noisy_attribute, T = T_est, p = data_eval['p_est'], CLIP_FLAG = config.clip_vec)

        correct_dp_true_fine = dp_violation(y_pred=y_pred, attribute=noisy_attribute, T = T_true, p = data_eval['p_true'], CLIP_FLAG = config.clip_vec)

        error_est_fine = correct_dp_est_fine
        error_true_fine = correct_dp_true_fine


        print(f'noisy dp: {noisy_dp}')
        print(f'[GLOBAL] correct dp est: {correct_dp_est}')
        print(f'[GLOBAL] correct dp true: {correct_dp_true}')
        print(f'true dp: {true_dp}')
        print(f'[LOCAL] correct dp est: {correct_dp_est_fine}')
        print(f'[LOCAL] correct dp true: {correct_dp_true_fine}')

        print(f'-------------\n')

        error = [error_base, error_est, error_true, error_est_fine, error_true_fine]
        return error

    elif metric == 'eo':
        # without conditional independence
        T_est, p_est, T_true, p_true = [], [], [], []
        p_est_only_p = []
        p_true_only_p = []
        for y in np.unique(y_true):
            p_est_tt = []
            p_true_tt = []

            Pky = []
            for k in np.unique(y_pred):
                loc = (y_pred == k) & (y_true == y)
                T_est_tmp, p_est_tmp, T_true_tmp, p_true_tmp = get_T_p(config, noisy_attribute[loc], lr = 0.1, true_attribute = true_attribute[loc])
                T_est.append(T_est_tmp)
                p_est_only_p.append(p_est_tmp)
                p_est_tt.append(p_est_tmp.reshape(-1))  # P(A=a|f(X)=k,Y=y), forall a
                T_true.append(T_true_tmp)
                p_true_only_p.append(p_true_tmp)
                p_true_tt.append(p_true_tmp.reshape(-1))
                Pky.append(np.sum(loc) / len(y_true)) # P(f(X)=k,Y=y), forall k
            Py = np.sum(Pky) # P(Y=y)
            p_est += [(np.sum(np.array(p_est_tt) * np.array(Pky).reshape(-1,1),0).reshape(-1,1) / Py).tolist()]*len(np.unique(y_true))  
            p_true += [(np.sum(np.array(p_true_tt) * np.array(Pky).reshape(-1,1),0).reshape(-1,1) / Py).tolist()]*len(np.unique(y_true)) 


        noisy_eo = eo_violation(y_pred=y_pred, y_true=y_true, attribute=noisy_attribute[:,0].reshape(-1), CLIP_FLAG = config.clip_vec)  # only use the first noisy label
        correct_eo_est = eo_violation(y_pred=y_pred, y_true=y_true, attribute=noisy_attribute, T = data_eval['T_est'], p = p_est, CLIP_FLAG = config.clip_vec) # prev: data_eval['p_est']
        correct_eo_true = eo_violation(y_pred=y_pred, y_true=y_true, attribute=noisy_attribute, T = data_eval['T_true'], p = p_true, CLIP_FLAG = config.clip_vec) # data_eval['p_true']
        true_eo = eo_violation(y_pred=y_pred, y_true=y_true, attribute=data_eval['true_attribute'], CLIP_FLAG = config.clip_vec)
        
        error_base = noisy_eo 
        error_est = correct_eo_est 
        error_true = correct_eo_true 

        correct_eo_est_fine = eo_violation(y_pred=y_pred, y_true=y_true,  attribute=noisy_attribute, T = T_est, p = p_est, CLIP_FLAG = config.clip_vec)
        correct_eo_true_fine = eo_violation(y_pred=y_pred, y_true=y_true, attribute=noisy_attribute, T = T_true, p = p_true, CLIP_FLAG = config.clip_vec)

        error_est_fine = correct_eo_est_fine
        error_true_fine = correct_eo_true_fine

        print(f'noisy eo: {noisy_eo}')
        print(f'[Global] correct eo est: {correct_eo_est}')
        print(f'[Global] correct eo true: {correct_eo_true}')
        print(f'true eo: {true_eo}')
        print(f'[LOCAL] correct eo est: {correct_eo_est_fine}')
        print(f'[LOCAL] correct eo true: {correct_eo_true_fine}')
        print(f'-------------\n')
        error = [error_base, error_est, error_true, error_est_fine, error_true_fine]
        return error


     
def get_T_p(config, noisy_attribute, lr = 0.1, true_attribute = None):

    # Estimate T and P with HOC
    T_est, p_est = get_T_global_min(config, noisy_attribute, lr = lr)
    print(f'\n\n-----------------------------------------')
    print(f'Estimation finished!')
    # np.set_printoptions(precision=1)
    print(f'The estimated T (*100) is \n{np.round(T_est*100,1)}')
    print(f'The estimated p (*100) is \n{np.round(p_est*100,1)}')
    if true_attribute is not None:
        T_true, p_true = check_T(KINDS=config.num_classes, clean_label=true_attribute, noisy_label=noisy_attribute)
        print(f'T_inv: \nest: \n{np.linalg.inv(T_est)}\ntrue:\n{np.linalg.inv(T_true)}')
        print(f'T_true: {T_true},\n T_est: {T_est}')
        print(f'p_true: {p_true},\n p_est: {p_est}')
    return T_est, p_est, T_true, p_true.reshape(-1,1)

if __name__ == "__main__":

    # Setup ------------------------------------------------------------------------

    config = parser.parse_args()
    config.device = set_device()
    CLIP_FLAG = config.clip_vec
    config.dataset_name = 'compas'
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    data = torch.load(config.data_path)
    # data format:
    #     data = {'train_pred': y_pred_train, 
    #             'val_pred': y_pred_val,
    #             'train_label': train_set.label,
    #             'val_label': val_set.label,
    #             'train_race': train_set.true_attribute,
    #             'val_race': val_set.true_attribute,
    #             'train_noisy_race':train_set.noisy_attribute,
    #             'val_noisy_race':val_set.noisy_attribute,
    #             'acc': [train_acc, test_acc],
    #             }
   
    noisy_attribute = data['train_noisy_race']
    true_attribute = data['train_race']
    y_true = data['train_label'].numpy()


    label_classes = np.unique(noisy_attribute.reshape(-1))
    print(f'Current label classes (sensitive attribute): {label_classes}')
    if np.min(label_classes) > 0:
        label_classes = np.min(label_classes)
        label_classes = label_classes.astype(int)
        print(f'Reset counting from 0')
        print(f'Current label classes: {label_classes}')
        
    config.num_classes = len(label_classes)
    
    T_est, p_est, T_true, p_true = get_T_p(config, noisy_attribute, lr = 0.1, true_attribute = true_attribute)


    # we only consider two race following convention. 
    # Our method is easily extended to multi-race.

    method_list = ['tree', 'forest', 'boosting', 'SVM','logit','nn', 'compas_score']

    error_rec = {'eo_est':[], 'eo_true':[], 'dp_est':[], 'dp_true':[],
                'eo_est_fine':[], 'eo_true_fine':[], 'dp_est_fine':[], 'dp_true_fine':[],
                'dp_base':[], 'eo_base':[]    }

    for method in method_list:
        config.data_path = f'./COMPAS/results/compas_ce_{method}.pt'
        data = torch.load(config.data_path)
        y_pred = data['train_pred']

        data_eval = {'y_pred': y_pred,
            'y_true': y_true,
            'noisy_attribute': noisy_attribute,
            'true_attribute': true_attribute,
            'T_est': T_est,
            'p_est': p_est,
            'T_true': T_true,
            'p_true': p_true }

        error_dp = eval_fairness(config, method, data_eval, metric='dp')

        error_rec['dp_base'].append(error_dp[0])
        error_rec['dp_est'].append(error_dp[1])
        error_rec['dp_true'].append(error_dp[2])
        error_rec['dp_est_fine'].append(error_dp[3])
        error_rec['dp_true_fine'].append(error_dp[4])


        error_eo = eval_fairness(config, method, data_eval, metric='eo')
        error_rec['eo_base'].append(error_eo[0])
        error_rec['eo_est'].append(error_eo[1])
        error_rec['eo_true'].append(error_eo[2]) 
        error_rec['eo_est_fine'].append(error_eo[3])
        error_rec['eo_true_fine'].append(error_eo[4])


    print('\n\n')

    df = pd.DataFrame.from_dict(error_rec) 
    df.to_csv(f'./result/error_rec_compas_{CLIP_FLAG}.csv', index = False, header=True)
    # print(improvement)
        


