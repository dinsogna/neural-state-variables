import time
import os
import sys
import json
import glob
import torch
import itertools
import numpy as np
from PIL import Image
from scipy import misc
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import datasets, transforms

class NeuralPhysDataset(Dataset):
    def __init__(self, data_filepath, flag, seed, object_name="double_pendulum"):
        self.seed = seed
        self.flag = flag
        self.object_name = object_name
        self.data_filepath = data_filepath
        self.all_filelist = self.get_all_filelist()
        self.video_batch = True

    def get_all_filelist(self):
        filelist = []
        obj_filepath = os.path.join(self.data_filepath, self.object_name)
        # get the video ids based on training or testing data
        with open(os.path.join('../datainfo', self.object_name, f'data_split_dict_{self.seed}.json'), 'r') as file:
            seq_dict = json.load(file)
        vid_list = seq_dict[self.flag]

        # go through all the selected videos and get the triplets: input(t, t+1), output(t+2)
        for vid_idx in vid_list:
            seq_filepath = os.path.join(obj_filepath, str(vid_idx))
            num_frames = len(os.listdir(seq_filepath))
            suf = os.listdir(seq_filepath)[0].split('.')[-1]
            for p_frame in range(num_frames - 3):
                par_list = []
                for p in range(4):
                    par_list.append(os.path.join(seq_filepath, str(p_frame + p) + '.' + suf))
                filelist.append(par_list)
        return filelist

    def __len__(self):
        if self.video_batch:
            return len(self.all_filelist) // 57
        return len(self.all_filelist)

    # 0, 1 -> 2, 3
    def __getitem__(self, idx):
        if self.video_batch:
            file_list = self.all_filelist[57 * idx : 57 * (idx+1)]
            data, target, filepath = [], [], []
            for par_list in file_list:
                tmp_data, tmp_target = [], []
                for i in range(2):
                    tmp_data.append(self.get_data(par_list[i])) # 0, 1
                tmp_data = torch.cat(tmp_data, 2)
                data.append(tmp_data)
                tmp_target.append(self.get_data(par_list[-2])) # 2
                tmp_target.append(self.get_data(par_list[-1])) # 3
                tmp_target = torch.cat(tmp_target, 2)
                target.append(tmp_target)
                filepath.append('_'.join(par_list[0].split('/')[-2:]))
            data = torch.stack(data)
            target = torch.stack(target)
            return data, target, filepath

        par_list = self.all_filelist[idx]
        data = []
        for i in range(2):
            data.append(self.get_data(par_list[i])) # 0, 1
        data = torch.cat(data, 2)
        target = []
        target.append(self.get_data(par_list[-2])) # 2
        target.append(self.get_data(par_list[-1])) # 3
        target = torch.cat(target, 2)
        filepath = '_'.join(par_list[0].split('/')[-2:])
        return data, target, filepath

    def get_data(self, filepath):
        data = Image.open(filepath)
        data = data.resize((128, 128))
        data = np.array(data)
        data = torch.tensor(data / 255.0)
        data = data.permute(2, 0, 1).float()
        return data

    def set_video_batch(video_batch_bool):
        self.video_batch = video_batch_bool

class NeuralPhysRefineDataset(TensorDataset):
    def __init__(self, data, target, filepaths):
        self.data = data
        self.target = target
        self.filepaths = filepaths
        self.video_batch = True

    def __len__(self):
        if self.video_batch:
            return len(self.filepaths) // 57
        return len(self.filepaths)

    # 0, 1 -> 2, 3
    def __getitem__(self, idx):
        if self.video_batch:
            data = self.data[57 * idx : 57 * (idx + 1)]
            target = self.target[57 * idx : 57 * (idx + 1)]
            filepath = [f'{int(f[0])}_{int(f[1])}.png' for f in self.filepaths[57 * idx : 57 * (idx + 1)]]
            # filepath = np.array(filepath, dtype=int)
            return data, target, filepath

        data = self.data[idx]
        target = self.target[idx]
        filepath = '_'.join(self.filepaths[idx])
        return data, target, filepath

    def set_video_batch(video_batch_bool):
        self.video_batch = video_batch_bool


class NeuralPhysLatentDynamicsDataset(Dataset):
    def __init__(self, data_filepath, flag, seed, object_name="double_pendulum"):
        self.seed = seed
        self.flag = flag
        self.object_name = object_name
        self.data_filepath = data_filepath
        self.all_filelist = self.get_all_filelist()

    def get_all_filelist(self):
        filelist = []
        obj_filepath = os.path.join(self.data_filepath, self.object_name)
        # get the video ids based on training or testing data
        with open(os.path.join('../datainfo', self.object_name, f'data_split_dict_{self.seed}.json'), 'r') as file:
            seq_dict = json.load(file)
        vid_list = seq_dict[self.flag]

        # go through all the selected videos and get the triplets: input(t, t+1), output(t+2)
        for vid_idx in vid_list:
            seq_filepath = os.path.join(obj_filepath, str(vid_idx))
            num_frames = len(os.listdir(seq_filepath))
            suf = os.listdir(seq_filepath)[0].split('.')[-1]
            for p_frame in range(num_frames - 5):
                par_list = []
                for p in range(6):
                    par_list.append(os.path.join(seq_filepath, str(p_frame + p) + '.' + suf))
                filelist.append(par_list)
        return filelist

    def __len__(self):
        return len(self.all_filelist)

    # 0, 1 -> 2, 3
    def __getitem__(self, idx):
        par_list = self.all_filelist[idx]
        data = []
        for i in range(2):
            data.append(self.get_data(par_list[i])) # 0, 1
        data = torch.cat(data, 2)
        target = []
        target.append(self.get_data(par_list[2])) # 2
        target.append(self.get_data(par_list[3])) # 3
        target = torch.cat(target, 2)
        target_target = []
        target_target.append(self.get_data(par_list[-2])) # 4
        target_target.append(self.get_data(par_list[-1])) # 5
        target_target = torch.cat(target_target, 2)
        filepath = '_'.join(par_list[0].split('/')[-2:])
        return data, target, target_target, filepath

    def get_data(self, filepath):
        data = Image.open(filepath)
        data = data.resize((128, 128))
        data = np.array(data)
        data = torch.tensor(data / 255.0)
        data = data.permute(2, 0, 1).float()
        return data