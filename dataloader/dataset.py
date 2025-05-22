import torch
import torch.utils.data as data
import numpy as np
import os
import pickle as pkl
from tqdm import tqdm
import json
from hydra.utils import to_absolute_path
    

''' Keystroke, Hand Pose, and ground truth EMG dataset for training and testing'''
class KeyPoseEmgDataset(data.Dataset):

    def __init__(self, mode, win_len, overlap_len, pose_win_len, pose_overlap, data_root, dataset_split=None, infer_performance=None):
        self.mode = mode
        self.win_len, self.pose_win_len= win_len, pose_win_len
        if self.mode == "test" or self.mode == "val":
            self.overlap_len, self.pose_overlap = 0, 0
        else:
            self.overlap_len, self.pose_overlap = overlap_len, pose_overlap

        self.data_root = data_root
        self.data = []
        error_files = []

        if infer_performance is not None:
            self.load_file(infer_performance, error_files)

        else:
            dataset_split_path = dataset_split
            with open(to_absolute_path(dataset_split_path), 'r') as f:
                dataset_all_list = json.load(f)
            self.dataset_list = dataset_all_list[self.mode]
            self.basename_list = [os.path.splitext(file)[0] for file in self.dataset_list]
            print(f"Total {len(self.basename_list)} performances in {self.mode} mode, loading dataset...")

            for file in tqdm(self.basename_list):
                self.load_file(file, error_files)
            print(f"{len(error_files)} error files")


    def load_file(self, file, error_files):
        with open(to_absolute_path(f'{self.data_root}/keystroke_data/{file}.pkl'), 'rb') as f2:
            keystroke_data = pkl.load(f2)
        keystroke_data = torch.tensor(keystroke_data).float()
        with open(to_absolute_path(f'{self.data_root}/emg_data/{file}.pkl'), 'rb') as f:
            emg_data = pkl.load(f)
        emg_data = torch.tensor(emg_data).float()
        if emg_data.shape[0] != keystroke_data.shape[0]:
            raise ValueError("Error! {file}'s EMG frames number is not equal to keystroke") 
         
        # with open(f'{self.data_root}/hand_data/movie0/smoothed_keyps/{file}.pkl', 'rb') as f3:
        with open(to_absolute_path(f'{self.data_root}/hand_data/keyp_top/{file}.pkl'), 'rb') as f3:
            pose_data = pkl.load(f3)
        # with open(f'{self.data_root}/hand_data/movie3/smoothed_keyps/{file}.pkl', 'rb') as f3:
        with open(to_absolute_path(f'{self.data_root}/hand_data/keyp_right/{file}.pkl'), 'rb') as f3:
            pose_data_right = pkl.load(f3)

        pose_data = torch.tensor(pose_data).float()[:,:,:2]# len, 21, 2
        pose_data = pose_data.reshape(pose_data.shape[0], -1) # len, 42
        pose_data_right = torch.tensor(pose_data_right).float()[:,:,:2] # len, 21, 2
        pose_data_right = pose_data_right.reshape(pose_data_right.shape[0], -1) # len, 42
    

        num_windows = (keystroke_data.shape[0] - self.overlap_len) // (self.win_len - self.overlap_len)
        num_windows_pose = (pose_data.shape[0] - self.pose_overlap) // (self.pose_win_len - self.pose_overlap)
        num_windows_min = min(num_windows, num_windows_pose)
        
        split_keystroke, num_windows_with_last1 = self.segment_sliding_subarray(keystroke_data, self.win_len, self.overlap_len, num_windows_min)
        split_emg, num_windows_with_last2 = self.segment_sliding_subarray(emg_data, self.win_len, self.overlap_len, num_windows_min)
        split_pose, num_windows_with_last3 = self.segment_sliding_subarray(pose_data, self.pose_win_len, self.pose_overlap, num_windows_min)
        split_pose_r, num_windows_with_last4 = self.segment_sliding_subarray(pose_data_right, self.pose_win_len, self.pose_overlap, num_windows_min)
        if not (num_windows_with_last1 == num_windows_with_last2 == num_windows_with_last3 == num_windows_with_last4):
            raise ValueError(f"Error! {file} has different number of windows with last")
        
        for i in range(num_windows_with_last1):
            self.data.append((split_keystroke[i], split_emg[i], split_pose[i], split_pose_r[i]))

    
    def segment_sliding_subarray(self, arr, win_len, overlap_len, num_windows):
        sub_arrs = []
        arr = arr.numpy()
        step_len = win_len - overlap_len
        num_windows_with_last = num_windows
        for i in range(num_windows):
            start_idx = i * step_len
            end_idx = start_idx + win_len
            sub_arrs.append(arr[start_idx:end_idx])
        return np.array(sub_arrs), num_windows_with_last

    
    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        data = self.data[idx] # is a tuple, like ([keystroke],[emg])
        return {
            "keystroke": data[0],
            "emg": data[1],
            "pose_t": data[2],
            "pose_r": data[3],
        }
    



class Emg2PoseDataset(data.Dataset):
    def __init__(self, mode, win_len, overlap_len, pose_win_len, pose_overlap, infer_performance=None):
        self.mode = mode
        self.win_len, self.pose_win_len= win_len, pose_win_len
        if self.mode == "test" or self.mode == "val":
            self.overlap_len, self.pose_overlap = 0, 0
        else:
            self.overlap_len, self.pose_overlap = overlap_len, pose_overlap

        self.data_root = '/home/Desktop/emg2pose_dataset'
        self.data = []
        error_files = []
        if infer_performance is not None:
            self.load_file(infer_performance, error_files)
        else:
            other_root = '/home/Desktop/Github/Piano_EMG_NIPS25/preprocess'
            # with open(f'{other_root}/train_6500_test_val_500_dataset_VQ.json', 'r') as f:
            # with open(f'{other_root}/train_test_val_dataset.json', 'r') as f:
            with open(f'{other_root}/emg2pose_dataset_3_min_test.json', 'r') as f:
                dataset_all_list = json.load(f)
            self.dataset_list = dataset_all_list[self.mode]
            self.basename_list = [os.path.splitext(file)[0] for file in self.dataset_list]
            print(f"Total {len(self.basename_list)} performances in {self.mode} mode, loading {self.mode} dataset...")
            for file in tqdm(self.basename_list):
                self.load_file(file, error_files)
            print(f"{len(error_files)} error files")

    def load_file(self, file, error_files):
        with open(f'{self.data_root}/emg_data/2_normalized_emg/{file}.pkl', 'rb') as f:
            emg_data = pkl.load(f)
        emg_data = torch.tensor(emg_data).float()
       
        with open(f'{self.data_root}/hand_data/joint_angles/{file}.pkl', 'rb') as f3:
            pose_data = pkl.load(f3)
        pose_data = torch.tensor(pose_data).float()# len, 788, 3
        pose_data = pose_data.reshape(pose_data.shape[0], -1) # len, 788*3
    
        num_windows = (emg_data.shape[0] - self.overlap_len) // (self.win_len - self.overlap_len)
        num_windows_pose = (pose_data.shape[0] - self.pose_overlap) // (self.pose_win_len - self.pose_overlap)
        num_windows_min = min(num_windows, num_windows_pose)
        
        split_emg, num_windows_with_last2 = self.segment_sliding_subarray(emg_data, self.win_len, self.overlap_len, num_windows_min)
        split_pose, num_windows_with_last3 = self.segment_sliding_subarray(pose_data, self.pose_win_len, self.pose_overlap, num_windows_min)
        if not (num_windows_with_last2 == num_windows_with_last3):
            raise ValueError(f"Error! {file} has different number of windows with last")
        
        for i in range(num_windows_with_last2):
            self.data.append((split_emg[i], split_pose[i]))

    
    def segment_sliding_subarray(self, arr, win_len, overlap_len, num_windows):
        sub_arrs = []
        arr = arr.numpy()
        step_len = win_len - overlap_len
        num_windows_with_last = num_windows
        for i in range(num_windows):
            start_idx = i * step_len
            end_idx = start_idx + win_len
            sub_arrs.append(arr[start_idx:end_idx])

        return np.array(sub_arrs), num_windows_with_last

    
    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        data = self.data[idx] # is a tuple, like ([keystroke],[emg])
        return {
            "emg": data[0],
            "pose": data[1],
        }
    


    
def create_dataloader(dataset, args, dataset_type, shuffle, batch_size):
    DATALOADER = {
        'KeyPoseEmgDataloader': KeyPoseEmgDataset,
    }

    loader_cls = dataset_type
    assert loader_cls in DATALOADER.keys(), f'DataLoader {loader_cls} does not exist.'
    loader = DATALOADER[loader_cls]

    dataloader = data.DataLoader(
        dataset         = dataset,
        batch_size      = batch_size,
        shuffle         = shuffle,
        num_workers     = args.num_threads,
        drop_last = False,
        pin_memory      = False,
        prefetch_factor = 2 if args.num_threads > 0 else None,
    )
    return dataloader
