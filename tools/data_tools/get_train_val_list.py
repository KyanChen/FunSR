import glob
import json
import os
import numpy as np

img_path = 'samples/UCMerced'
dataset_name = 'uc'  # uc or aid
save_folder = img_path + '/..'
os.makedirs(save_folder, exist_ok=True)

sub_folder_list = glob.glob(img_path +'/*')
train_val_frac = {'uc': [0.6, 0.2], 'aid': [0.8, 0.0]}[dataset_name]  # train, val

assert dataset_name in ['uc', 'aid']

train_list = []
val_list = []
test_list = []

for sub_folder in sub_folder_list:
    img_list = glob.glob(sub_folder+'/*')
    np.random.shuffle(img_list)

    if dataset_name == 'uc':
        num_train_samps = int(len(img_list) * train_val_frac[0])
        num_val_samps = int(len(img_list) * train_val_frac[1])
    elif dataset_name == 'aid':
        num_train_samps = int(len(img_list) * train_val_frac[0]) - 10
        num_val_samps = 10

    train_list += img_list[:num_train_samps]
    val_list += img_list[num_train_samps:num_train_samps+num_val_samps]
    test_list += img_list[num_train_samps+num_val_samps:]

data = {}
for phase in ['train_list', 'val_list', 'test_list']:
    data[phase.split('_')[0]] = [os.path.basename(os.path.dirname(file)) + '/' + os.path.basename(file) for file in eval(phase)]

json.dump(data, open(save_folder+f'/{dataset_name}_split.json', 'w'), indent=4)