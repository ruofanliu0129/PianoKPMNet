#########################################################
#  INFERENCE    
#  python inference.py
#  change checkpoint_path and checkpoint_name in ./config/base.yaml
#########################################################

import os
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
import pickle as pkl
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from collections.abc import Callable, Sequence
import json
from module.emg_modules import EmgModule
from config.options import PianoKPMOptions
from dataloader.dataset import  KeyPoseEmgDataset, create_dataloader
from module.loss import EmgLoss, cal_ot_loss, cal_mse_loss
from hydra.utils import to_absolute_path


# inference configurations
def get_configurations():
    parser = PianoKPMOptions()
    opt = parser.get_options(is_train=False)  # change is_train to train/inference
    
    opt.is_continue = False
    opt.use_pose = True
    opt.use_key = True

    opt.mode = "train"
    opt.epoch = 200
    opt.batch_size = 64
    opt.batch_size_val = 64
    opt.log_rate = 1
    opt.win_len = 1024
    opt.win_len_val = 1024
    opt.pose_win_len = 60
    opt.overlap = 256
    opt.pose_overlap = 15

    opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return opt, opt.device


def save_recon_emg(pred_data, file):
    save_dir = "inference_emg"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{file.split('.')[0]}_recon.pkl")
    with open(save_path, 'wb') as f:
        pkl.dump(pred_data.cpu().numpy(), f)


def plot_recon_emg(emg_data, pred_data, keystroke, file):
    seq_len, emg_channels, keystroke_channels = emg_data.shape[0], emg_data.shape[1], keystroke.shape[1]
    labels = ['ADM', 'PB', '1DI', '2DI', '3DI', '4DI']
    fig, axs = plt.subplots(3, 1, figsize=(15, 15))
    # Plot Keystroke Data
    for channel in range(keystroke_channels):
        if torch.mean(keystroke[:, channel]) != -1:
            axs[0].plot(range(seq_len), keystroke[:, channel].cpu().detach().numpy(), label=f'Key {channel+1}')
    axs[0].set_title('Keystroke')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Height')
    axs[0].legend()
    axs[0].set_ylim(-1, 1)

    # Plot Ground Truth EMG Data
    for channel in range(emg_channels):
        axs[1].plot(range(seq_len), emg_data[:, channel].cpu().detach().numpy(), label=labels[channel])
    axs[1].set_title('GT EMG')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Amplitude')
    axs[1].legend()
    axs[1].set_ylim(-1, 1)

    # Plot Predicted EMG Data
    for channel in range(emg_channels):
        axs[2].plot(range(seq_len), pred_data[:, channel].cpu().detach().numpy(), label=labels[channel])
    axs[2].set_title('Pred EMG')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Amplitude')
    axs[2].legend()
    axs[2].set_ylim(-1, 1)

    plt.tight_layout()
    
    save_dir = "inference_plots"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{file.split('.')[0]}_plot.png"))
    plt.close(fig)



instantiate = hydra._internal.instantiate._instantiate2.instantiate
def make_lightning_module(config: DictConfig):
    """Create lightning module from experiment config."""
    model: EmgModule = instantiate(config=config.emg_module,  _convert_="all") 
    return model



def test_one_batch(plot_count, model, infer_loader, opt, device, file):

    model.eval()
    gt_emgs_list, recon_emgs_list, gt_keystroke_list = [], [], []
    with torch.no_grad():
        for i, data in enumerate(tqdm(infer_loader, desc="Inferring", unit="batch")):
            data['keystroke'], data['emg'] = data['keystroke'].to(device), data['emg'].to(device)
            data['pose_t'], data['pose_r'] = data['pose_t'].to(device), data['pose_r'].to(device)
            gt_keystroke, gt_emg, gt_pose_t, gt_pose_r = data['keystroke'], data['emg'], data['pose_t'], data['pose_r']
            
            key_bct = data['keystroke'].permute(0,2,1) # B C T
            emg_bct = data['emg'].permute(0,2,1)
            pose_t_bct = data['pose_t'].permute(0,2,1)
            pose_r_bct = data['pose_r'].permute(0,2,1)
            
            if opt.use_pose and not opt.use_key:
                pianokpm_input = {
                    'model_input': torch.cat([pose_t_bct, pose_r_bct], dim=1), # B 84 60
                    'emg': emg_bct,
                    'temb': None
                }
            elif not opt.use_pose and opt.use_key:
                pianokpm_input = {
                    'model_input': key_bct, # B 88 1024
                    'emg': emg_bct,
                    'temb': None
                }
            elif opt.use_pose and opt.use_key:
                pianokpm_input = {
                    'model_input': torch.cat([pose_t_bct, pose_r_bct], dim=1), # B 84 60
                    'emg': emg_bct,
                    'temb': key_bct
                }

            out_emg = model.inference(pianokpm_input, provide_initial_pos=False)
            out_emg = out_emg.permute(0,2,1) # -> B 1024 6
            recon_emgs = out_emg

            gt_emgs_list.append(gt_emg)
            recon_emgs_list.append(recon_emgs)
            gt_keystroke_list.append(gt_keystroke)
            
        
        flattened_gt_emgs = torch.cat(gt_emgs_list, dim=0).reshape(-1, 6) 
        flattened_recon_emgs = torch.cat(recon_emgs_list, dim=0).reshape(-1, 6)
        flattened_gt_keystroke = torch.cat(gt_keystroke_list, dim=0).reshape(-1, 88)


        # save and plot the emg prediction
        if plot_count % opt.log_rate == 0:
            save_recon_emg(flattened_recon_emgs, file)
            plot_recon_emg(flattened_gt_emgs, flattened_recon_emgs, flattened_gt_keystroke, file)

        mse_loss = cal_mse_loss(flattened_gt_emgs, flattened_recon_emgs)
        ot_loss = cal_ot_loss(flattened_gt_emgs, flattened_recon_emgs)

    return mse_loss, ot_loss



def TEST(
    config: DictConfig,
    extra_callbacks: Sequence[Callable] | None = None,
):
     
    opt, device = get_configurations()

    model = make_lightning_module(config).to('cuda')
    checkpoint_path, checkpoint_name = config.checkpoint_path, config.checkpoint_name
    state_dict = torch.load(to_absolute_path(os.path.join(checkpoint_path, checkpoint_name)), map_location='cuda')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    print(f"loading checkpoint from {checkpoint_path}")
    model.load_state_dict(state_dict)

    mode = "test"
    with open(to_absolute_path(config.dataset_split), 'r') as f:
        dataset_all_list = json.load(f)
        dataset_list = dataset_all_list[mode]
    
    all_mse_loss, all_rmse_loss, all_ot_loss = [], [], []
    plot_count = 0
    for file_fullpath in tqdm(dataset_list):
        file = os.path.splitext(file_fullpath)[0]
        infer_dataset = KeyPoseEmgDataset(mode, win_len=opt.win_len_val, overlap_len=0, pose_win_len=opt.pose_win_len, pose_overlap=0, data_root=config.data_root, infer_performance=file)
        infer_loader = create_dataloader(infer_dataset, opt, "KeyPoseEmgDataloader", False, opt.batch_size_val)
       
        mse_loss, ot_loss = test_one_batch(plot_count, model, infer_loader, opt,  device, file)
        plot_count += 1
        logging.info(f"{file} -- mse_loss: {mse_loss:.4f}, rmse_loss: {np.sqrt(mse_loss):.4f}, ot_loss: {ot_loss:.4f}")

        all_mse_loss.append(mse_loss)
        all_rmse_loss.append(np.sqrt(mse_loss))
        all_ot_loss.append(ot_loss)

    mean_mse_loss, mean_rmse_loss, mean_ot_loss = np.mean(all_mse_loss), np.mean(all_rmse_loss), np.mean(all_ot_loss)

    logging.info(f"--------------------------------")
    logging.info(f"checkpoint: {checkpoint_path}/{checkpoint_name}")
    logging.info(f'tested {len(all_mse_loss)} performances:')
    logging.info(f"mean_mse_loss: {mean_mse_loss:.4f}, mean_rmse_loss: {mean_rmse_loss:.4f}, mean_ot_loss: {mean_ot_loss:.4f}")
 
    print(f"testing {len(all_mse_loss)} performances")
    print(f"checkpoint: {checkpoint_path}/{checkpoint_name}")
    print(f"mean_mse_loss: {mean_mse_loss:.4f}, mean_rmse_loss: {mean_rmse_loss:.4f},  mean_ot_loss: {mean_ot_loss:.4f}")

    


@hydra.main(config_path="config", config_name="base", version_base="1.1")
def main(config: DictConfig):
    TEST(config)



if __name__ == '__main__':

    main()
