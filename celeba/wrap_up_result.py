import torch
from jax import numpy as jnp
from experiments.data import load_celeba_dataset
from run_celeba import *
from experiments.models import get_model, get_apply_fn_test
from experiments.train_state import get_train_state
from experiments.train import test, get_test_step
from jax import jit
import numpy as np

import sys
sys.path.append("..")
from COMPAS.utils import check_T
from numpy.testing import assert_array_almost_equal
def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    # print (np.max(y), P.shape[0])
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    #print m
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = int(y[idx])
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


methods = ['opencv']  # use opencv by default. Accuracy is 0.9255439459127228
ATTR_KEY = "attributes"
IMAGE_KEY = "image"
LABEL_KEY = "Smiling"
GROUP_KEY = "Male"
BATCH_SIZE = 128



ds_train, ds_test = load_celeba_dataset(args, batch_size = args.test_batch_size)
for example in ds_test:
    args.image_shape = example[IMAGE_KEY].numpy().shape[1:]
    break

# inference
args.load_dir = './exps/exps/run_0'
# args.ckpt = '38610'
args.ckpt = '19500'
model = get_model(args)
state, args = get_train_state(args, model)
f_test = get_apply_fn_test(model)
test_step = jit(get_test_step(f_test))

group_rec, label_rec, pred_rec = test(test_step, state, ds_train, args.test_batch_size, detail=True)
train_acc = jnp.mean((jnp.array(label_rec) == jnp.array(pred_rec))*1.0)

group_noisy_rec = []
for method_i in methods:
    data_tmp = torch.load(f'gender_pred_{method_i}.pt')
    group_noisy = data_tmp['gender_pred']

    print(f'data loaded from gender_pred_{method_i}.pt')
    # print(jnp.mean((jnp.array(group_noisy) == jnp.array(group_rec))*1.0))
    group_noisy_rec.append(group_noisy)


exp_name = 'smile_gender'
save_path = f'{exp_name}.pt'
save_dict = {'train_pred': pred_rec, 
            'val_pred': None,
            'train_label': label_rec,
            'val_label': None,
            'train_gender': group_rec,
            'val_gender': None,
            'train_noisy_gender':group_noisy_rec.copy(),
            'val_noisy_gender': None,
            'acc': [train_acc, None],
            }

torch.save(save_dict, save_path)
print(f'Key words: {save_dict.keys()}\nResults saved to {save_path}\n')

error_rates = [[0.0, 0.0], [0.2,0.0], [0.2,0.2], [0.4,0.2], [0.4,0.4]]
for e1,e2 in error_rates:
    T = [[1-e1, e1],[e2, 1-e2]]
    noisy_attribute = multiclass_noisify(np.array(group_noisy_rec[0]), P=np.array(T),
                                            random_state=0)
    T_true, p_true = check_T(KINDS=2, clean_label=np.array(group_rec), noisy_label=noisy_attribute)
    print(T_true)
    print(p_true)
    save_dict['train_noisy_gender'] = [noisy_attribute.tolist()]

    save_path = f'{exp_name}_{e1}_{e2}.pt'
    torch.save(save_dict, save_path)
    print(f'Key words: {save_dict.keys()}\nResults saved to {save_path}\n')