import numpy as np
from .data import load_celeba_dataset
from .utils import set_global_seed
from deepface import DeepFace
import torch
import torch.nn.functional as F

# physical_gpus = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_gpus[0],True)
# logical_gpus = tf.config.list_logical_devices("GPU")

def cosDistance(features):
    # features: N*M matrix. N features, each features is M-dimension.
    
    features = F.normalize(features, dim=1) # each feature's l2-norm should be 1 
    similarity_matrix = torch.matmul(features, features.T)
    distance_matrix = 1.0 - similarity_matrix
    return distance_matrix

def pred_gender_2nn(args):
  # setup
  set_global_seed()
  ATTR_KEY = "attributes"
  IMAGE_KEY = "image"
  LABEL_KEY = "Smiling"
  GROUP_KEY = "Male"
  BATCH_SIZE = 128
  NUM_CLASSES = 1

  model_rep = [
    "Facenet", 
    "Facenet512", 
    "OpenFace", 
    "ArcFace", 
    "Dlib", 
    "SFace",
    ]

  data = torch.load(f'smile_gender_{args.e1}_{args.e2}.pt')
  group_noisy_rec = data['train_noisy_gender'][0]


  ds_train, ds_test = load_celeba_dataset(args, batch_size=1)

  knn_set_size = 32000
  embedding = []
  group_noisy_rec_new = []
  group_rec_tmp = []
  cnt = 0
  print(f'use model {model_rep[args.model_sel]}')


  for example in ds_train:

    # load a image
    image, _ = example[IMAGE_KEY].numpy(), example[ATTR_KEY][GROUP_KEY].numpy().astype(np.uint8)
    bgr = image[...,::-1].copy()[0]


    #embeddings
    tmp = DeepFace.represent(img_path = bgr, model_name = model_rep[args.model_sel], enforce_detection = False)
    embedding.append(tmp)
    group_rec_tmp.append(group_noisy_rec[cnt])
    cnt += 1
    
    if len(embedding) == knn_set_size:
      print(f'cnt = {cnt}')
      embedding = torch.tensor(embedding)
      dist = cosDistance(embedding)
      _, loc = dist.topk(3, largest=False)
      group_rec_tmp = torch.tensor(group_rec_tmp)
      
      group_noisy_rec_new += group_rec_tmp[loc].tolist()

      group_rec_tmp = []
      embedding = []
  if embedding:
      embedding = torch.tensor(embedding)
      dist = cosDistance(embedding)
      _, loc = dist.topk(3, largest=False)
      group_rec_tmp = torch.tensor(group_rec_tmp)    
      group_noisy_rec_new += group_rec_tmp[loc].tolist()
      
  data['train_noisy_gender'] = np.transpose(group_noisy_rec_new)


  exp_name = 'smile_gender'
  save_path = f'{exp_name}_{model_rep[args.model_sel]}_{args.e1}_{args.e2}.pt'


  torch.save(data, save_path)
  print(f'Key words: {data.keys()}\nResults saved to {save_path}\n')



