import numpy as np
import pandas as pd
np.set_printoptions(suppress=True, precision=4)

import argparse

# Options ----------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--type", type=str, default='NE', help="NE (Normalized Error), RE (Raw Error), RD (Raw Disparity), I (Improvement)")

def Convert(string):
    li = np.array(string.strip('[]').split()).astype(float).tolist()
    return li


def get_gap_offset_improvement_seed(tmp_err, true, noisy):
    result_tmp = np.zeros((len(tmp_err),3))
    true = true.reshape(-1)
    noisy = noisy.reshape(-1)
    result_tmp[:,0] = np.abs(np.array(tmp_err) - true) # abs gap
    # result_tmp[:,0] = np.abs(np.array(tmp_err)) # raw value
    result_tmp[:,1] = np.round(result_tmp[:,0] / true * 100,2) # normalized disparity %
    result_tmp[:,2] = np.round((1 - result_tmp[:,0] / np.abs(noisy-true))*100,2) # improvement
    return result_tmp

def get_raw_disparity_seed(tmp_err, true, noisy):
    result_tmp = np.zeros((len(tmp_err),3))
    true = true.reshape(-1)
    noisy = noisy.reshape(-1)
    # result_tmp[:,0] = np.abs(np.array(tmp_err) - true) # abs gap
    result_tmp[:,0] = np.abs(np.array(tmp_err)) # raw value
    result_tmp[:,1] = np.abs(np.array(tmp_err)) # placeholder
    result_tmp[:,2] = np.abs(np.array(tmp_err)) # placeholder
    return result_tmp

if __name__ == "__main__":
    noise_rates = [[0.0, 0.0], [0.2,0.0], [0.2,0.2], [0.4,0.2], [0.4,0.4]]
    config = parser.parse_args()
    if config.type == 'RD':
        get_gap_offset_improvement = get_raw_disparity_seed
    elif config.type in ['RE','NE','I']:
        get_gap_offset_improvement = get_gap_offset_improvement_seed
    else:
        raise NameError('Wrong evaluation metrics')
    methods = ['dp_est',
    'eo_est',
    'dp_est_fine',
    'eo_est_fine']
    result_all = []
    noise_cnt = 0
    soft_e_all = [[0.2970611630431300, 0.12340036,  0.05587907],  # 0.0 0.0
              [0.23320235976120657, 0.0963198,  0.04050681],  # 0.2 0.0
              [0.1806306631037638, 0.07076932, 0.03409911],  # 0.2 0.2
              [0.11994934361663417, 0.04604174, 0.02298369],  # 0.4 0.2
              [0.057034317620733244, 0.02624631, 0.01134968]]  # 0.4 0.4
    for noise_i in noise_rates:
        # prepare data
        soft_e = soft_e_all[noise_cnt]
        noise_cnt += 1
        data_load = pd.read_csv(f'result/error_rec_celeba_{noise_i[0]}_{noise_i[1]}_True.csv').to_dict()
        true_dp_dict  = data_load['dp_true_fine']
        true_dp = np.array([true_dp_dict[i] for i in true_dp_dict])
        noisy_dp_dict  = data_load['dp_base']
        noisy_dp = np.array([noisy_dp_dict[i] for i in noisy_dp_dict])
        eo_t_dict = data_load['eo_true_fine']
        eo_t = np.array([Convert(eo_t_dict[i]) for i in eo_t_dict])
        eo_n_dict = data_load['eo_base']
        eo_n = np.array([Convert(eo_n_dict[i]) for i in eo_n_dict])

        # print(f'true dp: {true_dp}, noisy dp {noisy_dp}, true eo: {eo_t}, noisy eo: {eo_n}')

        result = np.zeros((8,18+9+9))
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
                tmp_err = np.array([dt_error[i] for i in dt_error])
                result_tmp = get_gap_offset_improvement(tmp_err/2, true_dp/2, noisy_dp/2)
                result[:,cnt*3:cnt*3+3] = result_tmp
                cnt += 1

        eo_n_soft = np.ones((len(true_dp),2)) 


        noisy_dp_soft = np.ones(len(true_dp)) * soft_e[0]
        eo_n_soft[:,0] = eo_n_soft[:,0] * soft_e[1]
        eo_n_soft[:,1] = eo_n_soft[:,1] * soft_e[2]


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
        # print(result)

        if config.type == 'NE':
        # get normalized error (Table 8)
            result = np.delete(result,np.arange(0,len(result[0]),3), axis=1) # remove raw value
            result = np.delete(result,np.arange(1,len(result[0]),2), axis=1) # remove improvement
            idx_tmp = np.array([9, 6, 0, 3])
            idx = idx_tmp.tolist() + (idx_tmp+1).tolist() + (idx_tmp+2).tolist() 
            result = result[:,idx] # DP*4, EO*4, EOp*4. (Org, *, Soft, Base)
        
        elif config.type in ['RD','RE']:
            if config.type == 'RE':
                print('Printing LaTex Style Raw Error for COMPAS')
            else:
                print('Printing LaTex Style Raw Disparity for COMPAS')
            # get abs gap or raw disparity
            result = np.delete(result,np.arange(1,len(result[0]),3), axis=1) # remove offset
            result = np.delete(result,np.arange(1,len(result[0]),2), axis=1) # remove improvement
            idx_tmp = np.array([9, 6, 0, 3])
            idx = idx_tmp.tolist() + (idx_tmp+1).tolist() + (idx_tmp+2).tolist() 
            result = result[:,idx] # DP*4, EO*4, EOp*4. (Org, *, Soft, Base)
        
        else:
            # get improvement
            result = np.delete(result,np.arange(0,len(result[0]),3), axis=1) # remove raw disparity
            result = np.delete(result,np.arange(0,len(result[0]),2), axis=1) # remove offset
            idx_tmp = np.array([9, 6, 0, 3])
            idx = idx_tmp.tolist() + (idx_tmp+1).tolist() + (idx_tmp+2).tolist() 
            result = result[:,idx] # DP*4, EO*4, EOp*4. (Org, *, Soft, Base)


        result = np.delete(result,(3,4), axis=0)

        
        result_all.append(result)
    # print(result)

    model_tmp = [
        "Facenet", 
        "Facenet512", 
        "OpenFace", 
        "ArcFace", 
        "Dlib", 
        "SFace",
        ]
    if len(result_all) == 1:
        result = result_all[0]
        model_rep = model_tmp
    else:
        result = []
        model_rep = []
        for i in range(len(result_all[0])):
            for j in range(len(result_all)):
                result.append(result_all[j][i])
                model_rep.append(model_tmp[i] + f' {noise_rates[j]}')

    result = np.array(result)

    # print(result)
    
    
    data_print = []
    tmp = "\color{red}"
    for i in range(result.shape[0]):
        data_print.append([model_rep[i]])
        if config.type in ['RE','RD']:
            for j in range(result.shape[1]-1):
                data_print[-1].append(f'& {result[i][j]:0.4f}')
            data_print[-1].append(f'& {result[i][-1]:0.4f} \\')
        else:
            for j in range(result.shape[1]-1):
                data_print[-1].append(f'& {result[i][j]:0.2f}')
            data_print[-1].append(f'& {result[i][-1]:0.2f} \\')
        if (i+1) % 5 == 0:
            data_print[-1] = ' '.join(data_print[-1]) + ' \hline'
        else:
            data_print[-1] = ' '.join(data_print[-1])
    print(data_print)
       
