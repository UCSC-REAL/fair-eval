from flax.training import checkpoints, lr_schedule
from jax import jit, value_and_grad
from jax import numpy as jnp
import numpy as np
import time
from .data import  load_celeba_dataset
from .metrics import binary_correct, constraints, fairness, hinge_loss
from .models import get_apply_fn_test, get_apply_fn_train, get_model
from .recorder import init_recorder, record_ckpt, record_test_stats, record_train_stats, save_recorder
# from .test import get_test_step, test
from .train_state import TrainState, get_train_state
from .utils import make_dir, print_args, save_args, set_global_seed
import tensorflow as tf

########################################################################################################################
#  Getters
########################################################################################################################

ATTR_KEY = "attributes"
IMAGE_KEY = "image"
LABEL_KEY = "Smiling"
GROUP_KEY = "Male"
IMAGE_SIZE = 28
NUM_CLASSES = 1

def create_vitaly_learning_rate_schedule():
  def learning_rate(step):
    base_lr, top, total = 0.2, 4680, 31200
    if step <= top:
      lr = base_lr * step / top
    else:
      lr = base_lr - base_lr * (step - top) / (total - top)
    return lr
  return learning_rate


def get_lr_schedule(args):
  if args.lr_vitaly:
    lr = create_vitaly_learning_rate_schedule()
  elif args.decay_steps:
    lr_sched_steps = [[e, args.decay_factor**(i + 1)] for i, e in enumerate(args.decay_steps)]
    lr_ = lr_schedule.create_stepped_learning_rate_schedule(args.lr, 1, lr_sched_steps)
    lr = lambda step: lr_(step).item()
  else:
    lr = lr_schedule.create_constant_learning_rate_schedule(args.lr, args.steps_per_epoch)
  return lr


def get_loss_fn(f_train):
  def loss_fn(params, model_state, x, y, z):
    logits, model_state = f_train(params, model_state, x)
    loss = hinge_loss(logits, y)
    con = constraints(logits, z)
    acc = jnp.mean(binary_correct(logits, y))
    return loss + 1. * con, (acc, logits, model_state)
  return loss_fn




def get_train_step(loss_and_grad_fn):
  def train_step(state, x, y, z, lr):
    (loss, (acc, logits, model_state)), gradient = loss_and_grad_fn(state.optim.target, state.model, x, y, z)
    new_optim = state.optim.apply_gradient(gradient, learning_rate=lr)
    state = TrainState(optim=new_optim, model=model_state)
    return state, logits, loss, acc, gradient
  return train_step


def get_test_step(f_test):
  def test_step(state, x, y, z):
    logits = f_test(state.optim.target, state.model, x)
    loss = hinge_loss(logits, y)
    acc = jnp.mean(binary_correct(logits, y))
    pos, neg = fairness(logits, y, z)
    return loss, acc, pos, neg, logits
  return test_step


def test(test_step, state, ds_test, batch_size, detail = False):
  loss, acc, pos, neg, N = 0, 0, 0, 0, 0
  Zp, Zn = 0, 0
  group_rec = []
  label_rec = []
  pred_rec = []
  for example in ds_test:
    image, group, label = example[IMAGE_KEY].numpy().astype(np.float32)/255., example[ATTR_KEY][GROUP_KEY].numpy().astype(np.uint8), example[ATTR_KEY][LABEL_KEY].numpy().astype(np.uint8)
    # print(image.shape)
    image = tf.image.resize(tf.constant(image), [IMAGE_SIZE,IMAGE_SIZE]).numpy()
    # print(image.shape)
    group_rec += group.tolist()
    label_rec += label.tolist()
    # train step
    x = image
    y = label
    z = group
    n = batch_size
    Zp += jnp.sum(z > 0)
    Zn += jnp.sum(z == 0)

    step_loss, step_acc, step_pos, step_neg, logits = test_step(state, x, y, z)
    pred_rec += jnp.uint8(logits.squeeze()>0).tolist()
    loss += step_loss * n
    acc += step_acc * n
    pos += step_pos
    neg += step_neg
    N += n
    # print(z)
    # print(acc/N, pos / Zp, neg / Zn)
    # print(acc, N, pos, neg)
  loss, acc = loss / N, acc / N
  pos, neg = pos / Zp, neg / Zn
  if detail:
    return group_rec, label_rec, pred_rec
  else:
    return loss, acc, pos - neg


########################################################################################################################
#  Bookkeeping
########################################################################################################################

def _log_and_save_args(args):
  print('train args:')
  print_args(args)
  save_args(args, args.save_dir, verbose=True)


def _make_dirs(args):
  make_dir(args.save_dir)
  make_dir(args.save_dir + '/ckpts')


def _print_stats(t, T, t_incr, t_tot, lr, train_acc, train_loss, test_acc, test_disp, test_loss, init=False):
  prog = t / T * 100
  lr = '  init' if init else f'{lr:.4f}'
  train_acc = ' init' if init else f'{train_acc:.3f}'
  train_loss = ' init' if init else f'{train_loss:.4f}'
  print(f'{prog:6.2f}% | time: {t_incr:5.1f}s ({t_tot/60:5.1f}m) | step: {t:6d} |',
          f'lr: {lr} | train acc: {train_acc} | train loss: {train_loss} | test acc: {test_acc:.3f} | test loss: {test_loss:.4f}')


def _record_test(rec, t, T, t_prev, t_start, lr, train_acc, train_loss, test_acc, test_disp, test_loss, init=False):
  rec = record_test_stats(rec, t, test_loss, test_acc)
  t_now = time.time()
  t_incr, t_tot = t_now - t_prev, t_now - t_start
  _print_stats(t, T, t_incr, t_tot, lr, train_acc, train_loss, test_acc, test_disp, test_loss, init)
  return rec, t_now


def _save_checkpoint(save_dir, step, state, rec):
  checkpoints.save_checkpoint(save_dir + '/ckpts', state, step, keep=10000)
  rec = record_ckpt(rec, step)
  return rec


########################################################################################################################
#  Train
########################################################################################################################


def train(args):
  # setup
  set_global_seed()
  _make_dirs(args)

  ds_train, _ = load_celeba_dataset(args, shuffle_files=True, batch_size=args.train_batch_size)
  _, ds_test = load_celeba_dataset(args, shuffle_files=False, batch_size=args.test_batch_size)

  for example in ds_test:
    args.image_shape = example[IMAGE_KEY].numpy().shape[1:]
    break

  # setup
  model = get_model(args)
  state, args = get_train_state(args, model)
  f_train, f_test = get_apply_fn_train(model), get_apply_fn_test(model)
  test_step = jit(get_test_step(f_test))
  train_step = jit(get_train_step(value_and_grad(get_loss_fn(f_train), has_aux=True)))
  lr = get_lr_schedule(args)
  rec = init_recorder()

  # info
  _log_and_save_args(args)
  time_start = time.time()
  time_now = time_start
  print('train net...')


  for epoch_i in range(args.num_epochs):
    t = 0
    for example in ds_train:
      t += 1
      # load data
      image, group, label = example[IMAGE_KEY].numpy().astype(np.float32)/255., example[ATTR_KEY][GROUP_KEY].numpy().astype(np.uint8), example[ATTR_KEY][LABEL_KEY].numpy().astype(np.uint8)
      image = tf.image.resize(tf.constant(image), [IMAGE_SIZE,IMAGE_SIZE]).numpy()
      x = image
      y = label
      z = group

      # train
      state, logits, loss, acc, grad = train_step(state, x, y, z, lr(t))
      rec = record_train_stats(rec, t-1, loss.item(), acc.item(), lr(t))
      if t == args.EP_STEPS:
        break

    # test and log
    test_loss, test_acc, test_disp = test(test_step, state, ds_test, args.test_batch_size)
    rec, time_now = _record_test(rec, t*epoch_i, t*args.num_epochs, time_now, time_start, lr(t), acc, loss, test_acc, test_disp, test_loss)

    rec = _save_checkpoint(args.save_dir, t*epoch_i, state, rec)

  # wrap it up
  save_recorder(args.save_dir, rec)
  return test_acc, test_disp
