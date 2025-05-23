#########################################################
#  TRAIN    
#  python train.py
#  change data_root and dataset_split in ./config/base.yaml
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
from module.emg_modules import EmgModule
from config.options import PianoKPMOptions
from dataloader.dataset import  KeyPoseEmgDataset, create_dataloader
from module.loss import EmgLoss, cal_ot_loss, cal_mse_loss



loss_fn = EmgLoss(weights={
    'mse': 1.0,
    'ot': 1.0
})


# training configurations
def get_configurations():
    parser = PianoKPMOptions()
    opt = parser.get_options(is_train=True)  # change is_train to train/inference
    
    opt.is_continue = False
    opt.use_pose = True
    opt.use_key = True

    opt.win_len = 1024
    opt.win_len_val = 1024
    opt.pose_win_len = 60
    opt.overlap = 256
    opt.pose_overlap = 15

    opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return opt, opt.device



def plot_recon_emg(emg_data, pred_data, epoch, opt, mode):
    batch_size, seq_len, emg_channels = emg_data.shape[0], emg_data.shape[1], emg_data.shape[2]

    for batch_idx in range(min(batch_size, 10)):
        fig, axs = plt.subplots(2, 1, figsize=(15, 10))

        # Plot Ground Truth EMG Data
        for channel in range(emg_channels):
            axs[0].plot(range(seq_len), emg_data[batch_idx, :, channel].cpu().detach().numpy(), label=f'Muscle {channel+1}')
        axs[0].set_title('GT EMG')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Amplitude')
        axs[0].legend()
        axs[0].set_ylim(-1,1)

        # Plot Predicted EMG Data
        for channel in range(emg_channels):
            axs[1].plot(range(seq_len), pred_data[batch_idx, :, channel].cpu().detach().numpy(), label=f'Muscle {channel+1}')
        axs[1].set_title('Pred EMG')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Amplitude')
        axs[1].legend()
        axs[1].set_ylim(-1,1)    

        plt.tight_layout()
        if mode == "train":
            save_dir = os.path.join('train_plots', 'E%04d' % (epoch))
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_dir = os.path.join('eval_plots', 'E%04d' % (epoch))
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"Epoch_{epoch}-Sample_{batch_idx}.png"))
        plt.close(fig)



instantiate = hydra._internal.instantiate._instantiate2.instantiate
def make_lightning_module(config: DictConfig):
    """Create lightning module from experiment config."""
    model: EmgModule = instantiate(config=config.emg_module,  _convert_="all") 
    return model



def run_one_batch(mode,data, model, vae, loss_fn, device, opt, config, i, epoch):
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

    out_emg, _ = model.forward(pianokpm_input, provide_initial_pos=False)
    out_emg = out_emg.permute(0,2,1) # -> B 1024 6

    recon_emgs = out_emg

    total_loss, loss_dict = loss_fn(gt_emg, recon_emgs)


    if i % config.train.log_rate == 0 and epoch % 20 ==0:
        plot_recon_emg(gt_emg, recon_emgs,  epoch, opt, mode)
        
    return total_loss, loss_dict['mse']



def train_and_val(model, train_loader, val_loader, optimizer, loss_fn, scheduler, opt, config, epoch, device):

    model.train()
    total_loss,total_recon_loss = 0,0
    for i, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{config.train.epoch}", unit="batch")):

        optimizer.zero_grad()
        loss, recon_loss = run_one_batch('train',data, model, None, loss_fn, device, opt, config, i, epoch)
        
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.empty_cache()
    
    avg_train_loss = total_loss / len(train_loader)
    avg_train_recon_loss = total_recon_loss / len(train_loader)
    scheduler.step()

    model.eval()
    total_val_loss, total_val_recon_loss = 0,0
    with torch.no_grad():
        for i, data in enumerate(tqdm(val_loader, desc="Validating", unit="batch")):
            loss, recon_loss = run_one_batch('val',data, model, None, loss_fn, device, opt, config, i, epoch)
            
            total_val_loss += loss.item()
            total_val_recon_loss += recon_loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    avg_val_recon_loss = total_val_recon_loss / len(val_loader)

    return avg_train_loss, avg_val_loss



def TRAIN(
    config: DictConfig,
    extra_callbacks: Sequence[Callable] | None = None,
):
    opt, device = get_configurations()

    model = make_lightning_module(config).to('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr)
    scheduler = StepLR(optimizer, step_size=config.train.step_size, gamma=config.train.gamma) # after each batch before eval

    train_dataset = KeyPoseEmgDataset(mode="train", win_len=opt.win_len, overlap_len=opt.overlap, pose_win_len=opt.pose_win_len, pose_overlap=opt.pose_overlap, data_root=config.data_root, dataset_split=config.dataset_split)
    val_dataset = KeyPoseEmgDataset(mode="val", win_len=opt.win_len_val, overlap_len=0, pose_win_len=opt.pose_win_len, pose_overlap=0, data_root=config.data_root, dataset_split=config.dataset_split)
    train_loader = create_dataloader(train_dataset, opt, "KeyPoseEmgDataloader", True, config.train.batch_size)
    val_loader = create_dataloader(val_dataset, opt, "KeyPoseEmgDataloader", True, config.train.batch_size_val)

    best_val_loss = float('inf')
    for epoch in range(1, config.train.epoch+1):
        train_loss, val_loss = train_and_val(model, train_loader, val_loader, optimizer, loss_fn, scheduler, opt, config, epoch, device)
        print(f"Epoch {epoch}/{config.train.epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        logging.info(f"Epoch {epoch}/{config.train.epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f'Best model saved at epoch {epoch}, with val_loss {best_val_loss}')
            logging.info(f'Best model saved at epoch {epoch}, with val_loss {best_val_loss}')
            os.makedirs('model', exist_ok=True)
            torch.save(model.state_dict(), os.path.join('model', 'best_epoch_'+str(epoch)+'.pth'))
        else:
            print(f"Not saved at epoch {epoch}, current val_loss is {val_loss}")
        
        torch.save(model.state_dict(), os.path.join('model', 'latest_epoch.pth'))

    torch.save(model.state_dict(), os.path.join('model', 'final_epoch_'+str(epoch)+'.pth'))



@hydra.main(config_path="config", config_name="base", version_base="1.1")
def main(config: DictConfig):
    TRAIN(config)



if __name__ == '__main__':

    main()
