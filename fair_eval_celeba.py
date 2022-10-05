import torch
from hoc import *
import random
import numpy as np
import pandas as pd
from fair_eval import eval_fairness, get_T_p, parser


parser.add_argument('--e1', type=float, default=0.0)
parser.add_argument('--e2', type=float, default=0.0)


if __name__ == "__main__":

    # Setup ------------------------------------------------------------------------
    config = parser.parse_args()
    config.device = set_device()
    CLIP_FLAG = config.clip_vec
    method_list = [
        "Facenet", 
        "Facenet512", 
        "OpenFace", 
        "ArcFace", 
        "Dlib", 
        "SFace",
        ]

    config.data_path = f'./celeba/smile_gender_{method_list[0]}_{config.e1}_{config.e2}.pt'

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    data = torch.load(config.data_path)

    noisy_attribute = np.transpose(data['train_noisy_gender'])
    num_sample = noisy_attribute.shape[0]
    true_attribute = np.array(data['train_gender'])[:num_sample]
    y_pred = np.array(data['train_pred'])[:num_sample]
    y_true = np.array(data['train_label'])[:num_sample]


    label_classes = np.unique(noisy_attribute.reshape(-1))
    print(f'Current label classes (sensitive attribute): {label_classes}')
    if np.min(label_classes) > 0:
        label_classes = np.min(label_classes)
        label_classes = label_classes.astype(int)
        print(f'Reset counting from 0')
        print(f'Current label classes: {label_classes}')
        
    config.num_classes = len(label_classes)
    
    T_est, p_est, T_true, p_true = get_T_p(config, noisy_attribute, lr = 0.1, true_attribute = true_attribute)

    
    error_rec = {'eo_est':[], 'eo_true':[], 'dp_est':[], 'dp_true':[],
            'eo_est_fine':[], 'eo_true_fine':[], 'dp_est_fine':[], 'dp_true_fine':[],
             'dp_base':[], 'eo_base':[]    }

    # go through all representation extractors
    for method in method_list:
        config.data_path = f'./celeba/smile_gender_{method}_{config.e1}_{config.e2}.pt'
        data = torch.load(config.data_path)
        noisy_attribute = np.transpose(data['train_noisy_gender'])

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
    df.to_csv(f'./result/error_rec_celeba_{config.e1}_{config.e2}_{CLIP_FLAG}.csv', index = False, header=True)

        


