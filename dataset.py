import numpy as np
import pandas as pd
import os

def data_select():

    train_data = pd.read_csv('train_data.csv')
    test_data = pd.read_csv('valid_AAF.csv')

    age_min, age_max = train_data['age'].min(), train_data['age'].max()

    age_group = [[int(age_min), 28], [24, 33], [29, 40], [34, 49], [41, int(age_max)]]

    sampling = False
    sample_rate = 0.50

    return train_data, test_data, age_group,  sample_rate