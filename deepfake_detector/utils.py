import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import subprocess
import pandas as pd
import numpy as np
from tqdm import tqdm
import re

METADATA_PATH= os.getcwd() + '/deepfake_detector/data/metadata/'

def vidtimit_setup_real_videos(path_to_dataset):
    """
    Setting up the vidtimit dataset of real videos. The path for all real videos should be: ./vidtimitreal
    All videos should be in unzipped in their respective folders e.g. ./vidtimitreal/fadg0/
    The videos from the vidtimit.csv must be downloaded separately from http://conradsanderson.id.au/vidtimit/
    """
    # add jpg extension to all files
    for path, dirs, files in os.walk(path_to_dataset):
        for filename in files:
            # give idx as new name, to avoid similar filenames in different folders
            os.rename(path + '/' + filename, path + '/' + filename + '.jpg')
    print("Creating .avi videos.")
    # create videos from jpgs
    file_ending = ".avi"
    counter = 0
    os.chdir(path_to_dataset)
    for path, dirs, files in os.walk(path_to_dataset):
        if path == path_to_dataset:
            continue
        for d in dirs:
            if d != 'video':
                vid_path = os.path.join(path + '/' + d + '/%03d.jpg')
                vid_path2 = os.path.join(path + '/' + d + '/')
                # create real videos in avi format similar to deepfakes and avoid duplicate names with counter
                subprocess.call(
                    ['ffmpeg', '-r', '25', '-i', f"{vid_path}", "-c:v", "libxvid", f'{vid_path2 + str(counter)+  file_ending}'])
                counter += 1
    # remove all jpgs, so that only the videos are left
    for path, dirs, files in os.walk(path_to_dataset):
        for filename in files:
            if filename.endswith(".jpg"):
                os.remove(path + '/' + filename)


def dfdc_metadata_setup():
    """Returns training, testing and validation video meta data frames for the DFDC dataset."""
    #read in metadata
    print("Reading metadata...this can take a minute.")
    meta_df_list=[]
    for file in tqdm(os.listdir(METADATA_PATH)):
        temp=pd.read_json(os.path.join(METADATA_PATH,file),orient='index')
        temp["folder"] = int(re.search("metadata(.*)\.json",file).group(1))
        meta_df_list.append(temp)

    all_meta = pd.concat(meta_df_list)
    all_meta['video'] = all_meta.index  # create video column from index
    all_meta.reset_index(drop=True, inplace=True)  # drop index
    all_meta.drop(['split'],axis=1, inplace=True)
    # recode labels
    all_meta['label'] = all_meta['label'].apply(
        lambda x: 0 if x == 'REAL' else 1)
    all_meta.drop(['original'],axis=1, inplace=True)
    # sample 16974 fakes from 45 folders -> that's approx. 378 fakes per folder
    train_df = all_meta[all_meta['folder'] < 45]
    # 16974 reals in train data and 89629 fakes
    reals = train_df[train_df['label'] == 0]
    #del reals['folder']
    reals['folder']
    fakes = train_df[train_df['label'] == 1]
    fakes_sampled = fakes[fakes['folder'] == 0].sample(378, random_state=24)
    # sample the same number of fake videos from every folder
    for num in range(45):
        if num == 0:
            continue
        sample = fakes[fakes['folder'] == num].sample(378, random_state=24)
        fakes_sampled = fakes_sampled.append(sample, ignore_index=True)
    # drop 36 videos randomly to have exactly 16974 fakes
    np.random.seed(24)
    drop_indices = np.random.choice(fakes_sampled.index, 36, replace=False)
    fakes_sampled = fakes_sampled.drop(drop_indices)
    #del fakes_sampled['folder']
    fakes_sampled['folder']
    all_meta_train = pd.concat([reals, fakes_sampled], ignore_index=True)
    # get 1000 samples from training data that are used for margin and augmentation validation
    real_sample = all_meta_train[all_meta_train['label'] == 0].sample(
        300, random_state=24)
    fake_sample = all_meta_train[all_meta_train['label'] == 1].sample(
        300, random_state=24)
    full_margin_aug_val = real_sample.append(fake_sample, ignore_index=True)
    # create test set
    test_df = all_meta[all_meta['folder'] > 44]
    all_meta_test = test_df.reset_index(drop=True)

    return all_meta_train, all_meta_test, full_margin_aug_val
