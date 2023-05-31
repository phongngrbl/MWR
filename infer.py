import argparse
from prediction import *
from dataset import *

if __name__ == '__main__':
    praser = argparse.ArgumentParser()
    args = praser.parse_args()
    args.backbone = "Global_Regressor"
    args.im_path = r'/home/rb025/RabilooAI/Age_Estimation/data/data'
    train_data, test_data, reg_bound, sample_rate = data_select()
    global_regression(args, train_data, test_data, sampling=False, sample_rate=sample_rate)
