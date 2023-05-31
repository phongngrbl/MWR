from utils import *
import network as models
import os
import pandas as pd
import torch
import argparse
from dataset import *
import time
import psutil


def global_regression(arg, train_data, test_data, sampling=False, sample_rate=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    ### Create network ###
    model = models.create_model(arg, arg.backbone)
    model.to(device)

    initial_model = 'mae_Epoch_39_MAE_5.4122_CS_0.6271.pth'

    print(initial_model)

    ### Load network parameters ###
    checkpoint = torch.load(initial_model, map_location=device)
    model_dict = model.state_dict()

    model_dict.update(checkpoint['model_state_dict'])
    model.load_state_dict(model_dict)
    print("=> loaded checkpoint '{}'".format(initial_model))

    model.eval()

    ### Get features ###
    #features = feature_extraction_global_regression(arg, train_data, test_data, model, device)
    start_time = time.time()
    start_memory = psutil.virtual_memory().used
    ### Make initial prediction ###
    features_train = np.load('features_train_1.npy', allow_pickle=True)
    features_test = np.load('features_test.npy', allow_pickle=True)
    init_pred = initial_prediction(train_data, test_data, features_train, features_test, 5)
    end_time = time.time()
    end_memory = psutil.virtual_memory().used
    excution_time = end_time - start_time
    memory_usage = end_memory - start_memory
    print(f"Time: {excution_time}")
    print(f"Memory: {memory_usage}")


    loss_total, pair_idx = get_pairs(arg, train_data,  features_train, features_test, init_pred, model, device, False)
    # np.save('loss_total.npy', loss_total)
    print('Best pair indices saved', 'loss_total.npy')

    ### Get best reference pairs ###
    refer_idx, refer_idx_pair = select_reference_global_regression(train_data, pair_idx, np.array(loss_total), limit=1)

    ### MWR ### PREDICT
    pred = MWR_global_regression(arg, train_data, test_data, features_train, features_test, refer_idx, refer_idx_pair,
                                 init_pred, model, device)





    # save_results_path = '/home/rb025/Documents/MWR/inference/results'
    # if os.path.isfile(save_results_path) == False:
    #     np.savetxt(save_results_path, np.array(pred).reshape(-1, 1))
    #     print('Global regression results saved', save_results_path)
    #
    # ### Viz results ###
    # get_results(arg, test_data, np.array(pred))



if __name__ == '__main__':
    praser = argparse.ArgumentParser()
    args = praser.parse_args()
    args.backbone = "Global_Regressor"
    #args.im_path = r'/home/rb025/RabilooAI/Age_Estimation/data/data'
    args.tau = 0.5
    train_data, test_data, reg_bound, sample_rate = data_select()
    global_regression(args, train_data, test_data, sampling=False, sample_rate=sample_rate)
