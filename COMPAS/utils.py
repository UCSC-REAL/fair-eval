# Some resources:
# https://dalex.drwhy.ai/python-dalex-fairness2.html
# https://www.rdocumentation.org/packages/fairml/versions/0.6.3/topics/compas

import pandas as pd
from datetime import datetime
import numpy as np
# ----------  Name --> Race ---------
from ethnicolr import census_ln # do not use this one 
from ethnicolr import pred_census_ln,pred_fl_reg_name_five_cat,pred_nc_reg_name
# from ethnicolr import pred_wiki_ln, pred_wiki_name
# from ethnicolr import pred_fl_reg_ln, pred_fl_reg_name, pred_fl_reg_ln_five_cat, pred_fl_reg_name_five_cat
# from ethnicolr import pred_nc_reg_name
# https://github.com/appeler/ethnicolr
 
 
import torch
import random
from torch.nn import functional as F

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def date_from_str(s):
    return datetime.strptime(s, '%Y-%m-%d %H:%M:%S')

def preprocess_compas(filename = "compas-scores-two-years.csv"):
# Pre-process the compas data
# Input: Raw COMPAS data (filename)
# Output: N*d pd.DataFrame 
#         N: number of instances 
#         d: feature dimension
# sensitive attribute: race, sex
# model prediction: decile_score
# ground-truth label: is_recid or two_year_recid

    raw_data = pd.read_csv(f'./COMPAS/{filename}')
    print(f'Num rows (total): {len(raw_data)}')

    # remove missing data
    df = raw_data[((raw_data['days_b_screening_arrest'] <=30) & 
        (raw_data['days_b_screening_arrest'] >= -30) &
        (raw_data['is_recid'] != -1) &
        (raw_data['c_charge_degree'] != 'O') & 
        (raw_data['score_text'] != 'N/A')
        )]
    # length of staying in jail
    df['length_of_stay'] = (df['c_jail_out'].apply(date_from_str) - df['c_jail_in'].apply(date_from_str)).dt.total_seconds()


    sel_columns = [ 'first', 'last',  'name',
                    'age',  'age_cat', 'race',  'sex',  
                    'decile_score', 'score_text', 'v_decile_score',  
                    'is_recid', 'two_year_recid', 
                    'priors_count', 'days_b_screening_arrest','length_of_stay', 'c_charge_degree']

    df = df[sel_columns]
    print(f'Num rows after filtering: {len(df)}\n')
    # print(df.info())
    # print(df.race.value_counts())
    # print(pd.crosstab(df.race, df.is_recid))

    return df




class LogisticRegression(torch.nn.Module):
     def __init__(self, in_dim = 11, out_dim = 2):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)
     def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

class TwoLayerNN(torch.nn.Module):
     def __init__(self, in_dim = 11, out_dim = 5, hidden_size = 64, batch_norm = True):
        super(TwoLayerNN, self).__init__()
        if batch_norm:
            self.layer1 = torch.nn.Sequential(torch.nn.Linear(in_dim, hidden_size), torch.nn.BatchNorm1d(hidden_size), torch.nn.ReLU(True))
        else:
            self.layer1 = torch.nn.Sequential(torch.nn.Linear(in_dim, hidden_size), torch.nn.ReLU(True))
        self.layer2 = torch.nn.Sequential(torch.nn.Linear(hidden_size, out_dim))
     def forward(self, x):
        x = self.layer1(x)
        y_pred = self.layer2(x)
        return y_pred


# Adjust learning rate and for SGD Optimizer
def adjust_learning_rate(optimizer, epoch,alpha_plan):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]

def name2race(df, func=census_ln, num_race = 4):
    del df['race']
    print(f'Use function {func.__name__} to generate noisy attributes')
    if func.__name__ == 'census_ln':
        result_census = func(df, 'last', year=2010)
        
    elif 'census' in func.__name__:
        result_census = func(df, 'last', year=2010)

    elif '_ln' in func.__name__:
        result_census = func(df, 'last')
    else:
        del df['name']
        result_census = func(df, 'last', 'first')
    result_census = race_encode(result_census, func, num_race = num_race)
    # print(result_census.race.value_counts())
    return result_census

def race_encode(df, func = None, num_race = 4):
    race_pred = np.zeros((len(df),4)) # KEEP IT FIXED! only consider the case with not more than 4 races
    if func is None:
        func_encode = race_encode_compas
        print('Before encoding:')
        print(df.race.value_counts())
        df.race = df.race.apply(func_encode)
        print('After encoding:')
        print(df.race.value_counts())
        print('\n')


    elif func.__name__ == 'census_ln':
        df.replace('(S)','0.0', inplace = True)
        result_tmp = df[['pctblack', 'pctwhite',  'pcthispanic', 'pctapi', 'pctaian', 'pct2prace']].astype(float).to_numpy()
        if num_race == 4:
            result_tmp[:,1] /= 3.0  
            pred_race = np.argmax(result_tmp, axis = 1)
            pred_race[pred_race>=3] = 3
        elif num_race == 2:
            result_tmp[:,1] /= 3.0  
            pred_race = np.argmax(result_tmp, axis = 1)
            pred_race[pred_race>=1] = 1


        
        df['race'] = pred_race

    elif 'census' in func.__name__:
        func_encode = race_encode_census
        print(df.columns[15:-1])
        for i_race in df.columns[15:-1]:
            race_pred[:,func_encode(i_race)] += df[i_race].astype(float).to_numpy()
        if num_race == 4:
            race_pred[:,1] /= 4.0
            # pass  
        elif num_race == 2:
            race_pred[:,1] /= 4.0  
        pred_race = np.argmax(race_pred, axis = 1)
        race_pred[:,1] = np.max(race_pred[:,1:],axis=1)
        race_pred[:,:2] /= np.sum(race_pred[:,:2],axis=1).reshape(-1,1)
        df['race_pred'] = race_pred[:,0]
        df['race'] = pred_race
    elif 'wiki' in func.__name__:
        func_encode = race_encode_wiki
        print(df.columns[15:-1])
        for i_race in df.columns[15:-1]:
            race_pred[:,func_encode(i_race)] += df[i_race].astype(float).to_numpy()
        if num_race == 4:
            race_pred[:,1] /= 30.0  
            race_pred[:,0] *= 5.0  
        elif num_race == 2:
            race_pred[:,0] *= 15.0    

        pred_race = np.argmax(race_pred, axis = 1)
        race_pred[:,1] = np.max(race_pred[:,1:],axis=1)
        race_pred[:,:2] /= np.sum(race_pred[:,:2],axis=1).reshape(-1,1)
        df['race_pred'] = race_pred[:,0]
        df['race'] = pred_race
    elif 'fl_reg' in func.__name__:
        func_encode = race_encode_fl_reg
        print(df.columns[15:-1])
        for i_race in df.columns[15:-1]:
            race_pred[:,func_encode(i_race)] += df[i_race].astype(float).to_numpy()
        if num_race == 4:
            race_pred[:,1] *= 1.6
            race_pred[:,3] *= 2
        pred_race = np.argmax(race_pred, axis = 1)
        race_pred[:,1] = np.max(race_pred[:,1:],axis=1)
        race_pred[:,:2] /= np.sum(race_pred[:,:2],axis=1).reshape(-1,1)
        df['race_pred'] = race_pred[:,0]
        df['race'] = pred_race
    elif 'nc_reg' in func.__name__:
        func_encode = race_encode_nc_reg
        print(df.columns[15:-1])
        for i_race in df.columns[15:-1]:
            race_pred[:,func_encode(i_race)] += df[i_race].astype(float).to_numpy()
        if num_race == 4:
            race_pred[:,0] *= 1.5
            race_pred[:,1] *= 1.5
            race_pred[:,2] /= 4
            race_pred[:,3] /= 1.5
        elif num_race == 2:
            race_pred[:,0] *= 3

        pred_race = np.argmax(race_pred, axis = 1)
        race_pred[:,1] = np.max(race_pred[:,1:],axis=1)
        race_pred[:,:2] /= np.sum(race_pred[:,:2],axis=1).reshape(-1,1)
        df['race_pred'] = race_pred[:,0]
        df['race'] = pred_race
    return df


def race_encode_compas(s):
# COMPAS mapping: (combine 3 and 4 due to the sample size)
# Race                  #
# African-American    3175
# Caucasian           2103
# Hispanic             509
# Other                343
# Asian                 31
# Native American       11
# African-American          --> Black       --> 0
# Caucasian                 --> White       --> 1
# Hispanic                  --> Hispanic    --> 2
# Asian                     --> Asian       --> 3
# Other, Native American    --> Other       --> 4
    race_dict = {'African-American':0,'Caucasian':1, 'Hispanic':2}
    return race_dict.get(s, 3)

def race_encode_census(s):
# census mapping:
# black       -->   Black
# white       -->   White
# hispanic    -->   Hispanic
# api         -->   Asian
    race_dict = {'black':0,'white':1, 'hispanic':2}
    return race_dict.get(s, 3)



def race_encode_wiki(s):
# wiki mapping:
# GreaterEuropean,British                  --> White
# GreaterEuropean,WestEuropean,Hispanic    --> Hispanic
# GreaterEuropean,WestEuropean,Italian     --> White
# GreaterEuropean,WestEuropean,French      --> White
# GreaterEuropean,Jewish                   --> White
# Asian,GreaterEastAsian,EastAsian         --> Asian
# GreaterAfrican,Muslim                    --> Black
# Asian,GreaterEastAsian,Japanese          --> Asian
# Asian,IndianSubContinent                 --> Asian
# GreaterEuropean,WestEuropean,Germanic    --> White
# GreaterAfrican,Africans                  --> Black
# GreaterEuropean,EastEuropean             --> White
# GreaterEuropean,WestEuropean,Nordic      --> White
    race_dict = {'GreaterAfrican,Muslim':0, 
                'GreaterAfrican,Africans': 0, 
                'GreaterEuropean,British':1, 
                'GreaterEuropean,WestEuropean,Italian':1,
                'GreaterEuropean,WestEuropean,French':1,
                'GreaterEuropean,Jewish':1,
                'GreaterEuropean,WestEuropean,Germanic':1,
                'GreaterEuropean,EastEuropean':1,
                'GreaterEuropean,WestEuropean,Nordic':1,
                'GreaterEuropean,WestEuropean,Hispanic':2}
    return race_dict.get(s, 3)


def race_encode_fl_reg(s):
# fl_reg five_cat mapping:
# nh_black    -->   Black
# nh_white    -->   White
# hispanic    -->   Hispanic
# asian       -->   Asian
# other       -->   Other
    race_dict = {'nh_black': 0, 
                'nh_white':1,
                'hispanic':2}
    return race_dict.get(s, 3)




def race_encode_nc_reg(s):
# NL+B    -->   Black
# NL+W    -->   White
# HL+O    -->   Hispanic
# HL+W    -->   Hispanic
# NL+M    -->   Other
# NL+O    -->   Other
# NL+I    -->   Asian
# NL+A    -->   Asian
# HL+M    -->   Hispanic
# HL+B    -->   Hispanic
# HL+I    -->   Hispanic
# HL+A    -->   Hispanic
    race_dict = {'NL+B': 0, 
                'NL+W':1,
                'HL+O':2, 'HL+W':2, 'HL+M':2, 'HL+B':2, 'HL+I':2, 'HL+A':2}
    return race_dict.get(s, 3)

def race_encode_plain(s):
    return s

def check_T(KINDS, clean_label, noisy_label):
    T_real = np.zeros((KINDS,KINDS))
    if len(noisy_label.shape) > 1:
        num_noisy_label = noisy_label.shape[1]
    else:
        num_noisy_label = 1
        noisy_label = noisy_label.reshape(-1,1)
    for agent_i in range(num_noisy_label):
        for i in range(clean_label.shape[0]):
            T_real[clean_label[i]][noisy_label[i][agent_i]] += 1
    P_real = [sum(T_real[i])*1.0 for i in range(KINDS)] # random selection
    for i in range(KINDS):
        if P_real[i]>0:
            T_real[i] /= P_real[i]
    P_real = np.array(P_real)/sum(P_real)
    # print(f'Check: \nP = {P_real},\n T = \n{np.round(T_real,3)}')
    return T_real, P_real




def accuracy(logit, target, topk=(1,), loc = None):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    if loc is not None:
        correct = correct * loc
        batch_size = np.sum(loc)
    
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

