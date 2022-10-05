import numpy as np
import pandas as pd
from table_celeba import Convert, get_gap_offset_improvement_seed, get_raw_disparity_seed, parser
np.set_printoptions(suppress=True, precision=4)

if __name__ == "__main__":

    methods = ['dp_est', 'eo_est', 'dp_est_fine', 'eo_est_fine']
    config = parser.parse_args()
    if config.type == 'RD':
        get_gap_offset_improvement = get_raw_disparity_seed
    elif config.type in ['RE','NE','I']:
        get_gap_offset_improvement = get_gap_offset_improvement_seed
    else:
        raise NameError('Wrong evaluation metrics')


    # prepare data
    data_load = pd.read_csv('result/error_rec_compas_False.csv').to_dict()
    true_dp_dict  = data_load['dp_true_fine']
    true_dp = np.array([true_dp_dict[i] for i in true_dp_dict])
    noisy_dp_dict  = data_load['dp_base']
    noisy_dp = np.array([noisy_dp_dict[i] for i in noisy_dp_dict])
    eo_t_dict = data_load['eo_true_fine']
    eo_t = np.array([Convert(eo_t_dict[i]) for i in eo_t_dict])
    eo_n_dict = data_load['eo_base']
    eo_n = np.array([Convert(eo_n_dict[i]) for i in eo_n_dict])
    # print(f'true dp: {true_dp}, noisy dp {noisy_dp}, true eo: {eo_t}, noisy eo: {eo_n}')


    data_load_soft = pd.read_csv('result/error_rec_compas_False_soft_True_1000.csv').to_dict()
    noisy_dp_dict_soft  = data_load_soft['dp_base']
    noisy_dp_soft = np.array([noisy_dp_dict_soft[i] for i in noisy_dp_dict_soft])
    eo_n_dict_soft = data_load_soft['eo_base']
    eo_n_soft = np.array([Convert(eo_n_dict_soft[i]) for i in eo_n_dict_soft])




    # dp/2, eo/4，eop
    result = np.zeros((7,18+9+9))
    cnt = 0
    for method_i in methods:

        dt_error = data_load[method_i]

        if isinstance(dt_error[0], str):

            tmp_err = np.array([Convert(dt_error[i]) for i in dt_error])

            result_tmp = get_gap_offset_improvement(tmp_err[:,0].reshape(-1)/4, eo_t[:,0]/4, eo_n[:,0]/4)
            result[:,cnt*3:cnt*3+3] = result_tmp # equal odds
            cnt += 1
            result_tmp = get_gap_offset_improvement(tmp_err[:,1].reshape(-1), eo_t[:,1], eo_n[:,1])
            result[:,cnt*3:cnt*3+3] = result_tmp # equal opportunity
            cnt += 1


        else:
            # tmp_imp = [dt_improve[i] for i in dt_improve]
            tmp_err = np.array([dt_error[i] for i in dt_error])
            result_tmp = get_gap_offset_improvement(tmp_err/2, true_dp/2, noisy_dp/2)
            result[:,cnt*3:cnt*3+3] = result_tmp
            cnt += 1

    result[:,cnt*3:cnt*3+3] = get_gap_offset_improvement(noisy_dp_soft/2, true_dp/2, noisy_dp/2)
    cnt += 1
    result[:,cnt*3:cnt*3+3] = get_gap_offset_improvement(eo_n_soft[:,0].reshape(-1)/4, eo_t[:,0]/4, eo_n[:,0]/4)
    cnt += 1
    result[:,cnt*3:cnt*3+3] = get_gap_offset_improvement(eo_n_soft[:,1].reshape(-1), eo_t[:,1], eo_n[:,1])

    cnt += 1
    result[:,cnt*3:cnt*3+3] = get_gap_offset_improvement(noisy_dp/2, true_dp/2, noisy_dp/2)
    cnt += 1
    result[:,cnt*3:cnt*3+3] = get_gap_offset_improvement(eo_n[:,0].reshape(-1)/4, eo_t[:,0]/4, eo_n[:,0]/4)
    cnt += 1
    result[:,cnt*3:cnt*3+3] = get_gap_offset_improvement(eo_n[:,1].reshape(-1), eo_t[:,1], eo_n[:,1])
    # print(result) # DP, EO, EOp, DP*, EO*, EOp*, Soft (DP, EO, EOp), Base (DP, EO, EOp)

    if config.type == 'NE':
        # get normalized error (Table 6, Normalized Error)
        print('Printing LaTex Style Normalized Error for COMPAS')
        result = np.delete(result,np.arange(0,len(result[0]),3), axis=1) # remove raw value
        result = np.delete(result,np.arange(1,len(result[0]),2), axis=1) # remove improvement
        idx_tmp = np.array([9, 6, 0, 3])
        idx = idx_tmp.tolist() + (idx_tmp+1).tolist() + (idx_tmp+2).tolist() 
        result = result[:,idx] 

    elif config.type in ['RD','RE']:
        if config.type == 'RE':
            print('Printing LaTex Style Raw Error for COMPAS')
        else:
            print('Printing LaTex Style Raw Disparity for COMPAS')
        # get abs gap or raw disparity (Table 6, Raw Disparity or Raw Error)
        result = np.delete(result,np.arange(1,len(result[0]),3), axis=1) # remove offset
        result = np.delete(result,np.arange(1,len(result[0]),2), axis=1) # remove improvement
        idx_tmp = np.array([9, 6, 0, 3])
        idx = idx_tmp.tolist() + (idx_tmp+1).tolist() + (idx_tmp+2).tolist() 
        result = result[:,idx] 
    else:
        print('Printing LaTex Style Improvement for COMPAS')
        # get improvement
        result = np.delete(result,np.arange(0,len(result[0]),3), axis=1) # remove raw disparity
        result = np.delete(result,np.arange(0,len(result[0]),2), axis=1) # remove offset
        idx_tmp = np.array([9, 6, 0, 3])
        idx = idx_tmp.tolist() + (idx_tmp+1).tolist() + (idx_tmp+2).tolist() 
        result = result[:,idx] 

    # print(result)
    model_rep = ['tree', 'forest', 'boosting', 'SVM','logit','nn', 'compas_score']
    data_print = []
    for i in range(result.shape[0]):
        data_print.append([model_rep[i]])
        if config.type in ['RD','RE']:
            for j in range(result.shape[1]-1):
                data_print[-1].append(f'& {result[i][j]:0.4f}')
            data_print[-1].append(f'& {result[i][-1]:0.4f} \\')
            data_print[-1] = ' '.join(data_print[-1])
        else:
            for j in range(result.shape[1]-1):
                data_print[-1].append(f'& {result[i][j]:0.2f}')
            data_print[-1].append(f'& {result[i][-1]:0.2f} \\')
            data_print[-1] = ' '.join(data_print[-1])


    print(data_print)
    
