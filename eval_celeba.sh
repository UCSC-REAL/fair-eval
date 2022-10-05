# python3 fair_eval_celeba.py --e1 0.0 --e2 0.0 > eval_celeba_0.0_0.0_nc.log &
# python3 fair_eval_celeba.py --e1 0.2 --e2 0.0 > eval_celeba_0.2_0.0_nc.log &
# python3 fair_eval_celeba.py --e1 0.2 --e2 0.2 > eval_celeba_0.2_0.2_nc.log &
# python3 fair_eval_celeba.py --e1 0.4 --e2 0.2 > eval_celeba_0.4_0.2_nc.log &
# python3 fair_eval_celeba.py --e1 0.4 --e2 0.4 > eval_celeba_0.4_0.4_nc.log &
# wait

python3 fair_eval_celeba.py --e1 0.0 --e2 0.0 --clip_vec > eval_celeba_0.0_0.0.log 
python3 fair_eval_celeba.py --e1 0.2 --e2 0.0 --clip_vec > eval_celeba_0.2_0.0.log 
python3 fair_eval_celeba.py --e1 0.2 --e2 0.2 --clip_vec > eval_celeba_0.2_0.2.log
python3 fair_eval_celeba.py --e1 0.4 --e2 0.2 --clip_vec > eval_celeba_0.4_0.2.log 
python3 fair_eval_celeba.py --e1 0.4 --e2 0.4 --clip_vec > eval_celeba_0.4_0.4.log