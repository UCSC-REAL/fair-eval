from experiments import train
import argparse
# Options ----------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model_gender', type=str, default='opencv', help="retinaface mtcnn opencv ssd dlib")
parser.add_argument('--model_sel', type=int, default=1, help="VGG-Face, Facenet Facenet512,OpenFace, DeepFace, DeepID, ArcFace, Dlib, SFace")
parser.add_argument('--e1', type=float, default=0.0)
parser.add_argument('--e2', type=float, default=0.0)


# setup
ROOT = '.'
EXP = 'exps'
RUN = 0
META_MODEL_SEED, META_TRAIN_SEED, SEED_INCR = 42, 4242, 424242
EP_STEPS = 390
DATA_DIR = ROOT + '/data'
EXPS_DIR = ROOT + '/exps'

# arguments
# args = SimpleNamespace()
args = parser.parse_args()
# data
args.data_dir = DATA_DIR
args.dataset = 'celeba'

# model
args.model = 'resnet18_lowres'
args.model_seed = META_MODEL_SEED + RUN * SEED_INCR
args.load_dir = None
args.ckpt = 0
# optimizer
args.lr = 0.1
args.beta = 0.9
args.weight_decay = 0.0005
args.nesterov = True
args.lr_vitaly = False
args.decay_factor = 0.2
args.decay_steps = [50*EP_STEPS, 80*EP_STEPS, 90*EP_STEPS]
# training
args.num_epochs = 100
args.EP_STEPS = EP_STEPS
args.train_seed = META_TRAIN_SEED + RUN * SEED_INCR
args.train_batch_size = 128
args.test_batch_size = 1024
args.augment = True
args.track_forgetting = True
# checkpoints
args.save_dir = EXPS_DIR + f'/{EXP}/run_{RUN}'
args.log_steps = EP_STEPS
args.early_step = 0
args.early_save_steps = None
args.save_steps =  EP_STEPS
args.num_classes = 1
# experiment

if __name__ == "__main__":
    train(args)
