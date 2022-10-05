# from experiments.pred_gender_2nn import pred_gender_2nn
from experiments.pred_gender import pred_gender
from run_celeba import *

if __name__ == "__main__":

    # pred_gender, wrap_up_result, then pred_gender_2nn.
    pred_gender(args)


    # print(f'run with {args.model_sel} {args.e1} {args.e2}')
    # pred_gender_2nn(args)

    
