import numpy as np
from .data import load_celeba_dataset
from .utils import set_global_seed
from deepface import DeepFace
import torch




def pred_gender(args):
  # setup
  set_global_seed()
  ATTR_KEY = "attributes"
  IMAGE_KEY = "image"
  LABEL_KEY = "Smiling"
  GROUP_KEY = "Male"
  BATCH_SIZE = 128
  NUM_CLASSES = 1

  # models = [
  #   "VGG-Face", 
  #   "Facenet", 
  #   "Facenet512", 
  #   "OpenFace", 
  #   "DeepFace", 
  #   "DeepID", 
  #   "ArcFace", 
  #   "Dlib", 
  #   "SFace",
  #   ]
  models = [args.model_gender] # use opencv by default
  for model_i in models:
    gender_pred = []
    gender_pred_with_exception = []
    gender_pred_prob = []
    group_rec = []
    gender_map = {'Woman':0,'Man':1}
    args.batch_size = BATCH_SIZE
    ds_train, _ = load_celeba_dataset(args, batch_size = 1)
    cnt = 0
    num_exception = 0
    for example in ds_train:
      print(f'{model_i}: {cnt}/{len(ds_train)}')
      cnt += 1

      # get one image
      image, group = example[IMAGE_KEY].numpy(), example[ATTR_KEY][GROUP_KEY].numpy().astype(np.uint8)
      bgr = image[...,::-1].copy()[0]
      try:
        obj = DeepFace.analyze(img_path = bgr, actions = ['gender',], enforce_detection=True, prog_bar = False, detector_backend = model_i)
      except ValueError:
        print(f'face {cnt} not detected')
        num_exception += 1
        group_rec += group.tolist()
        gender_pred_with_exception.append(-1)
        obj = DeepFace.analyze(img_path = bgr, actions = ['gender'], enforce_detection=False, prog_bar = False, detector_backend = model_i)
        gender_pred.append(gender_map[obj['gender']])
        
      else:
        group_rec += group.tolist()
        gender_pred.append(gender_map[obj['gender']])
        gender_pred_with_exception.append(gender_map[obj['gender']])

      gender_pred_prob.append(obj['gender_prob_woman'])

      if cnt % 100 == 0:
        print(f'acc with exception: {np.mean(np.array(group_rec) == np.array(gender_pred_with_exception))}')
        print(f'acc without exception: {np.mean(np.array(group_rec) == np.array(gender_pred))}')
        print(gender_pred_prob[cnt-10:cnt])

    print(gender_pred)  
    print(group_rec)
    print(gender_pred_with_exception)
    torch.save({'gender_pred':gender_pred, 'group_rec':group_rec, 'gender_pred_with_exception':gender_pred_with_exception, 'gender_pred_prob':gender_pred_prob}, f'gender_pred_{model_i}.pt')
