from utils import *
from torch.utils.data import Dataset
import sklearn.preprocessing as preprocessing
import copy


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def input_dataset(name = 'compas', val_ratio = 0.2, seed = 0, func_list = None, num_race = 2, syn_noise = 0.2, num_labeler = 3, soft=False):
    if name == 'compas':
        # get compas data
        df = preprocess_compas()
        race_encode(df)
        dataset = CompasDataset(df.copy())
        dataset.generate_noisy_attribute(func_list=func_list, num_race=num_race, syn_noise=syn_noise, num_labeler = num_labeler)
        rand_idx = np.arange(len(dataset.label))
        setup_seed(seed)
        np.random.shuffle(rand_idx)
        thre = int(len(dataset.label) * val_ratio)

        if soft:
            prob = dataset.noisy_attribute_soft
            cnt = (prob * num_labeler).astype(int)
            noisy_attribute = np.ones((len(dataset.label),num_labeler))
            for i in range(len(dataset.label)):
                noisy_attribute[i][:cnt[i]] = 0
            idx = np.arange(num_labeler)
            np.random.shuffle(idx)
            noisy_attribute = noisy_attribute[:,idx]
            dataset.noisy_attribute = noisy_attribute

        # training dataset
        train_dataset = copy.copy(dataset)
        train_dataset.feature = dataset.feature[rand_idx[thre:]]
        train_dataset.label = dataset.label[rand_idx[thre:]]
        train_dataset.true_attribute = dataset.true_attribute[rand_idx[thre:]]
        train_dataset.score = dataset.score[rand_idx[thre:]]
        train_dataset.noisy_attribute = dataset.noisy_attribute[rand_idx[thre:]]

        # validation dataset --> to check fairness
        val_dataset = copy.copy(dataset)
        val_dataset.feature = dataset.feature[rand_idx[:thre]]
        val_dataset.label = dataset.label[rand_idx[:thre]]
        val_dataset.true_attribute = dataset.true_attribute[rand_idx[:thre]]
        val_dataset.score = dataset.score[rand_idx[:thre]]
        val_dataset.noisy_attribute = dataset.noisy_attribute[rand_idx[:thre]]

        print(f'training-validation split done. We have {len(train_dataset)} training instances and {len(val_dataset)} validation instances.')

        return train_dataset, val_dataset
    else:
        raise NameError('Undefined dataset')

class CompasDataset(Dataset):



    def __init__(self, data_file):
        FEATURES_CLASSIFICATION = ["age_cat", "sex", "priors_count", "c_charge_degree"] #features to be used for classification
        CONT_VARIABLES = ["priors_count"] # continuous features, will need to be handled separately from categorical features, categorical features will be encoded using one-hot
        CLASS_FEATURE = "two_year_recid" # the decision variable


        self.df = data_file.copy()
        data = data_file.to_dict('list')
        for k in data.keys():
            data[k] = np.array(data[k])


        Y = data[CLASS_FEATURE]
    
        X = np.array([]).reshape(len(Y), 0) # empty array with num rows same as num examples, will hstack the features to it

        feature_names = []
        for attr in FEATURES_CLASSIFICATION:
            vals = data[attr]
            if attr in CONT_VARIABLES:
                vals = [float(v) for v in vals]
                vals = preprocessing.scale(vals) # 0 mean and 1 variance
                vals = np.reshape(vals, (len(Y), -1)) # convert from 1-d arr to a 2-d arr with one col

            else: # for binary categorical variables, the label binarizer uses just one var instead of two
                lb = preprocessing.LabelBinarizer()
                lb.fit(vals)
                vals = lb.transform(vals)


            # add to learnable features
            X = np.hstack((X, vals))

            if attr in CONT_VARIABLES: # continuous feature, just append the name
                feature_names.append(attr)
            else: # categorical features
                if vals.shape[1] == 1: # binary features that passed through lib binarizer
                    feature_names.append(attr)
                else:
                    for k in lb.classes_: # non-binary categorical features, need to add the names for each cat
                        feature_names.append(attr + "_" + str(k))

        
        self.feature = torch.tensor(X, dtype=torch.float)
        self.label = torch.tensor(Y, dtype=torch.long)
        self.true_attribute = data['race']       
        self.score = torch.tensor(self.df.decile_score.to_list(), dtype=torch.long).view(-1,1)
        
        # rand_idx = np.arange(len(Y))
        # setup_seed(0)
        # np.random.shuffle(rand_idx)
        # thre = int(len(Y) * val_ratio)
        # self.feature_train = X[rand_idx[thre:]]
        # self.label_train = Y[rand_idx[thre:]]
        # self.true_attribute_train = data['race'][rand_idx[thre:]]
        # self.feature_val = X[rand_idx[:thre]]
        # self.label_val = Y[rand_idx[:thre]]
        # self.true_attribute_val = data['race'][rand_idx[:thre]]
        print(f'dataset construction done. \nShape of X {self.feature.shape}. \nShape of Y {self.label.shape}')
        # print(f'\nShape of validation X {self.feature_val.shape}. \nShape of validation Y {self.label_val.shape}')
        

    def generate_noisy_attribute(self, func_list = [], num_race = 2, syn_noise = 0.2, num_labeler = 3):
    # generate noisy attributes by external classifiers
    # Candidates:
    # * indicates the "good" classifiers
    # census_ln, pred_census_ln*
    # pred_wiki_ln, pred_wiki_name --> BAD. Training data contains few black sub-populations
    # pred_fl_reg_ln, pred_fl_reg_name, pred_fl_reg_ln_five_cat, pred_fl_reg_name_five_cat*
    # pred_nc_reg_name*
        candidate_external_classifiers = func_list
        
        clean_attribute = np.array(self.df.race).astype(int)
        clean_attribute[clean_attribute>=num_race-1] = num_race-1

        if func_list is None:
            print(f'External classifiers are missing. Synthesizing noisy attribute with noise rate: {syn_noise}')


            acc = 1-syn_noise
            std_acc = 0.05 if num_race > 2 else 0.2
            P_diag = acc + std_acc*2*(np.random.rand(num_race) - 0.5)
            # P_diag = np.linspace(min(upper_lower),max(upper_lower), num_race)
            if syn_noise < 0.001:
                P = np.eye(num_race)
            else:
                P = generate_noise_matrix_from_diagonal(diag = P_diag)
                P = np.array([[0.639, 0.361], [0.204, 0.796]])
            print(f'T_real is \n{P}')

            noisy_attribute = []
            for i in range(num_labeler):
                noisy_attribute.append(multiclass_noisify(clean_attribute, P=np.array(P),
                                            random_state=i))
            noisy_attribute = np.array(noisy_attribute).transpose()


        else:
            noisy_attribute = []
            noisy_attribute_soft = np.zeros(len(clean_attribute))
            for name2race_func in candidate_external_classifiers:
                df_pred = name2race(self.df.copy(), func=name2race_func, num_race=num_race)
                noisy_attribute.append(np.array(df_pred.race).astype(int))
                noisy_attribute_soft += np.array(df_pred.race_pred)
            noisy_attribute = np.array(noisy_attribute).transpose()
            noisy_attribute_soft /= len(candidate_external_classifiers) # only support binary
            


        # # three classes
        # clean_attribute[clean_attribute>=2] = 2
        # noisy_attribute[noisy_attribute>=2] = 2

        # two classes
        # race_pred[:,1] = np.max(race_pred[:,1:],axis=1)
        # df['race_pred'] = race_pred[:,:2] / np.sum(race_pred[:,:2], axis=1).reshape(-1,1) # only support binary race at present
        # clean_attribute[clean_attribute>=num_race-1] = num_race-1
        noisy_attribute[noisy_attribute>=num_race-1] = num_race-1

        # noisy_attribute = np.array(df_pred.race).astype(int)
        # num_attribute_classes = len(np.unique(clean_attribute))

        # print the true transition matrix
        T_real, P_real = check_T(KINDS=num_race, clean_label=clean_attribute, noisy_label=noisy_attribute)

        print('T*p')
        print(T_real * P_real.reshape(-1,1))
        self.noisy_attribute = noisy_attribute
        self.true_attribute = clean_attribute
        self.noisy_attribute_soft = noisy_attribute_soft

        # data_save = pd.DataFrame({'label_0':noisy_attribute[:,0],'label_1':noisy_attribute[:,1],'label_2':noisy_attribute[:,2]})

        # data_save.to_csv('compas_noisy_attribute_2race.csv',index=False,sep=',')

        # T_real, P_real = check_T(KINDS=num_attribute_classes, clean_label=clean_attribute, noisy_label=noisy_attribute[:,2])



    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        feature, label = self.feature[index], self.label[index]
        # feature[feature==99] = 0.0
        return feature, label, index
