# Evaluating Fairness Without Sensitive Attributes: A Framework Using Only Auxiliary Models 
(Official implementation)

# 1. COMPAS

Install required packages:
```shell
pip3 install torch
```

## Step 1: Get auxiliary models: Name --> race
Install the package [ethnicolr](https://github.com/appeler/ethnicolr):
```shell
pip3 install ethnicolr
```

## Step 2: Generate target models
Predict race with auxiliary models, then generate target models with 'tree', 'forest', 'boosting', 'SVM','logit','nn', and 'compas_score', respectively.

```shell
python3 COMPAS/train_classifier.py
```

## Step 3: Evaluate and calibrate fairness
```shell
python3 fair_eval.py
```

## Print results:
We've included our results in this repo.
```shell
# "NE (Normalized Error), RE (Raw Error), RD (Raw Disparity), I (Improvement)"
python3 table_compas.py --type NE
```

## 2. CelebA

Install required packages:
```shell
# CelebA is loaded with tensorflow dataset
pip3 install -q tfds-nightly tensorflow

# Install Jax and flax if you want to train the model by yourself
pip3 install flax==0.5.3 
pip3 install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html  

pip3 install torch
```


## Step 1: Get auxiliary models: DeepFace
Install the package [deepface](https://github.com/serengil/deepface):
```shell
pip3 install deepface
```

## Step 2: Generate target models
Generate a target model based on ResNet18

```shell
cd celeba
python3 run_celeba.py
```

Alternatively, you can just download our saved checkpoint from [this link](https://drive.google.com/file/d/1uAifyS7vb9vuMVTw74BDD-VjovB1ZL0B/view?usp=sharing) and save it to: ``./celeba/exps/exps/run_0/ckpts/checkpoint_19500``

## Step 3: Generate gender with auxiliary models
```shell
python3 gender_celeba_pred.py --model_gender opencv
```

## Step 4: Wrap up & 2NN
```shell
# Wrap up model predictions and sensitive attributes
python3 wrap_up_result.py

# find 2nn with different representation extractors
# model_rep = [
    # "Facenet", 
    # "Facenet512", 
    # "OpenFace", 
    # "ArcFace", 
    # "Dlib", 
    # "SFace",
    # ]
# (e1,e2) = [(0.0,0.0),
#            (0.2,0.0),
#            (0.2,0.2),
#            (0.4,0.2),
#            (0.4,0.4)]
python3 gender_celeba_2nn.py --model_sel 0 --e1 0.0 --e2 0.0
```

## Step 5: Evaluate and calibrate fairness
We have provided the preprocessed data for evaluating fairness. You can just run the last step to get results.

```shell
# (e1,e2) = [(0.0,0.0),
#            (0.2,0.0),
#            (0.2,0.2),
#            (0.4,0.2),
#            (0.4,0.4)]
python3 fair_eval_celeba.py --e1 0.0 --e2 0.0 --clip_vec
```
## Print results:
We've included our results in this repo.
```shell
# "NE (Normalized Error), RE (Raw Error), RD (Raw Disparity), I (Improvement)"
python3 table_celeba.py --type NE
```

**Note:** For Soft method, please refer to its original [paper](https://dl.acm.org/doi/10.1145/3287560.3287594).