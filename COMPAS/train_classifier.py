# Train classifier, generate noisy sensitive attribute, save all the results for fairness evaluation

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from dataset import *
from utils import *
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


import warnings
warnings.filterwarnings("ignore")




import argparse
parser = argparse.ArgumentParser(description='VA dataset analysis')
parser.add_argument('--epochs', default=20, type=int, help='number of total epochs to run')
parser.add_argument('--printfreq', default=20, type=int, help='print/test per printfreq epochs')
parser.add_argument('--hidden', default=128, type=int, help='size of hidden layer')
parser.add_argument('--balance', action='store_true', default=False)
parser.add_argument('--loss', default='ce', type=str, help="ce, bw, fw, peer")

args = parser.parse_args()
use_cuda = torch.cuda.is_available()

if __name__ == "__main__":
    
    dataset_name = 'compas'
    num_race = 2
    # syn_noise = 0.3
    candidate_external_classifiers = [pred_census_ln,pred_fl_reg_name_five_cat,pred_nc_reg_name]
    
    # num_labeler = 3 # [3, 5, 10, 20, 50, 100, 200, 500, 1000]
    # num_labeler = None
    # if num_labeler is not None:
    #     train_set, val_set = input_dataset(name = dataset_name, val_ratio = 0.1, seed = 0, func_list = None, num_race=num_race, syn_noise = syn_noise, num_labeler = num_labeler)
    # else:
    train_set, val_set = input_dataset(name = dataset_name, val_ratio = 0.1, seed = 0, func_list = candidate_external_classifiers, num_race=num_race)
    in_dim = train_set.feature.shape[1]
    args.num_classes = len(np.unique(train_set.label))
    args.methods = 'nn'
    batch_size = 128
    method_list = ['tree', 'forest', 'boosting', 'SVM','logit','nn', 'compas_score']
    # method_list = ['compas_score']
    for method in method_list:

        if method in ['tree', 'forest','boosting', 'SVM', 'compas_score']:
            if method == 'tree' or method == 'compas_score':
                clf = tree.DecisionTreeClassifier(criterion = 'entropy', min_impurity_decrease = 0.0007)
            elif method == 'forest':
                clf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', max_depth = 8, random_state  = 1)
            elif method == 'boosting':
                clf = AdaBoostClassifier(tree.DecisionTreeClassifier(criterion = 'entropy', min_impurity_decrease = 0.0007), n_estimators = 5, random_state  = 1, algorithm='SAMME', learning_rate = 1)
            elif method == 'SVM':
                clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))

            if method == 'compas_score':
                train_set.feature = train_set.score
                val_set.feature = val_set.score
          
            clf = clf.fit(train_set.feature, train_set.label)

            y_pred_train = clf.predict(train_set.feature)
            train_acc = (torch.sum(torch.tensor(y_pred_train)==train_set.label)/(len(train_set.label)*1.0)).item()
            y_pred_val = clf.predict(val_set.feature)


            numerator = torch.tensor(y_pred_val)==val_set.label
            denominator = len(val_set.label)*1.0
            test_acc = (torch.sum(numerator)/denominator).item()

            candidate_attribute = np.unique(val_set.true_attribute)
            test_acc_race = np.zeros(len(candidate_attribute))
            for i in candidate_attribute:
                loc = (val_set.true_attribute == i)
                test_acc_race[i] = (torch.sum(numerator * loc) / np.sum(loc)).item()
            

            print(f'train_acc {train_acc}, test_acc {test_acc}. \nper_race_test_acc: {test_acc_race}')
             

        else:
            # balanced training
            # weight = np.array([torch.sum(train_set.label==i).item() for i in range(args.num_classes)])

            # weight = 1.0 / weight
            # sample_weight = torch.tensor([weight[i] for i in train_set.label])
            # sampler = torch.utils.data.WeightedRandomSampler(sample_weight, len(sample_weight))
            # if use_cuda:
            #     val_set.feature = val_set.feature.cuda()
            #     val_set.label = val_set.label.cuda()
            #     train_set.feature = train_set.feature.cuda()
            #     train_set.label = train_set.label.cuda()
            # if args.balance:
            #     dataloader = DataLoader(train_set, batch_size=batch_size, sampler=sampler,
            #                     shuffle=False)
            # else:
            dataloader = DataLoader(train_set, batch_size=batch_size, 
                                shuffle=True)
            if method == 'logit':
                model = LogisticRegression(in_dim = in_dim, out_dim = args.num_classes)  # Logistic regression
            elif method == 'nn':
                model = TwoLayerNN(in_dim = in_dim, hidden_size = args.hidden, batch_norm = True, out_dim = args.num_classes)  # Two-Layer NN
                # model = ThreeLayerNN(in_dim = in_dim, hidden_size = [args.hidden1, args.hidden2], batch_norm = True, pretrained_path=args.pre_path)  # Three-Layer NN
            else:
                NameError('undefined method')
            if use_cuda:
                model = model.cuda()
                train_set.noise_prior = train_set.noise_prior.cuda()
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(),  weight_decay=1e-1)
            # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4, momentum=0.9)
            # optimizer = torch.optim.Adadelta(model.parameters(), weight_decay=1e-4)
            test_acc_max = 0.0
            # alpha_plan = [0.5] * 120 + [0.05] * 120 + [0.005] * 120
            # alpha_plan = [0.5] * 40 + [0.05] * 40 + [0.005] * 120
            for epoch in range(1,args.epochs+1):
                # adjust_learning_rate(optimizer, epoch, alpha_plan, beta_max=args.beta)
                for i_batch, (feature, label, index) in enumerate(dataloader):
                    model.train()
                    optimizer.zero_grad()

                    if use_cuda:
                        feature = feature.cuda()
                        label = label.cuda()

                    y_pred = model(feature)
                    # Compute Loss
                    if args.loss == 'ce':
                        loss = criterion(y_pred, label)
                    else:
                        raise NameError('Undefined loss function')
                    # Backward pass
                    loss.backward()
                    optimizer.step()

                if epoch % args.printfreq == 0:
                    model.eval()
                    
                    y_pred_val = model(val_set.feature)
                    test_acc_per_race = []
                    for i in np.unique(val_set.true_attribute):
                        loc = val_set.true_attribute == i
                        prec = accuracy(y_pred_val, val_set.label, topk=(1,), loc = loc)[0]
                        test_acc_per_race.append(prec.item())
                    prec = accuracy(y_pred_val, val_set.label, topk=(1,))[0]
                    test_acc = prec.item()
                    # test_acc_max = np.max((test_acc_max,test_acc))

                    y_pred_train = model(train_set.feature)
                    prec = accuracy(y_pred_train, train_set.label, topk=(1,))[0]
                    train_acc = prec.item()

                    print(f'[Epoch {epoch}. Train: {train_acc}. Test: {test_acc}', flush=True)
                    print(f'Per race test acc: {test_acc_per_race}')

                    y_pred_train = y_pred_train.argmax(1).cpu().numpy()
                    y_pred_val = y_pred_val.argmax(1).cpu().numpy()

        
        exp_name = dataset_name + '_'
        exp_name += args.loss + '_' + method
        print(f'[{exp_name}] val acc = {np.mean(np.array(test_acc))}')
        # if num_labeler is not None:
        #     save_path = f'./COMPAS/results/{exp_name}_check{num_labeler}.pt'
        # else:
        save_path = f'./COMPAS/results/{exp_name}.pt'
        save_dict = {'train_pred': y_pred_train, 
                    'val_pred': y_pred_val,
                    'train_label': train_set.label,
                    'val_label': val_set.label,
                    'train_race': train_set.true_attribute,
                    'val_race': val_set.true_attribute,
                    'train_noisy_race':train_set.noisy_attribute,
                    'val_noisy_race':val_set.noisy_attribute,
                    'acc': [train_acc, test_acc],
                    }
        torch.save(save_dict, save_path)
        print(f'Key words: {save_dict.keys()}\nResults saved to {save_path}\n')

