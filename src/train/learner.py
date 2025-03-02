from models.prepare_model import get_diffusion
from data.dataloader import get_train_val_dataloaders
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter

POLYFFUSION_CKPT_PATH = "/root/autodl-tmp/multipoly/polyffusion_ckpts/ldm_chd8bar/sdf+pop909wm_mix16_chd8bar/01-11_102022/chkpts/weights_best.pt"
CHORD_ENCODER_CKPT_PATH = "/root/autodl-tmp/multipoly/pretrained/chd8bar/weights.pt"
TRAINABLE_DICT = {"n_intertrack_head":4, "num_intertrack_encoder_layers":1,"intertrack_attention_levels":[2,3]}
DATAFOLDER = "/root/autodl-tmp/multipoly/data/lmd/lpd_5_midi/"   
BATCH_SIZE = 7
NUM_WORKERS = 4                         
LR = 1e-7

class Learner:
    def __init__(self, output_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.diffusion = get_diffusion(POLYFFUSION_CKPT_PATH, CHORD_ENCODER_CKPT_PATH, TRAINABLE_DICT, freeze_polyffusion=False).to(self.device)
        self.train_loader, self.val_loader = get_train_val_dataloaders(DATAFOLDER, BATCH_SIZE, NUM_WORKERS, train_ratio=0.95, pin_memory=True)
        
        self.optimizer = optim.Adam(self.diffusion.parameters(), lr=LR)

        self.output_dir = output_dir
        self.log_dir = self.output_dir + "/logs"
        self.ckpt_dir = self.output_dir + "/ckpts"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.autocast = torch.cuda.amp.autocast(enabled=True)

        self.writer = SummaryWriter(self.log_dir)
        self.epoch = 0
        self.step = 0

        self.log_train_loss_interval = 1000
        self.validation_interval = 20000
        self.best_val_loss = 1e10
        
        
    def train(self):
        while True:
            running_loss = []
            from tqdm import tqdm
            for batch in tqdm(self.train_loader):
                
                self.diffusion.train()
                multi_prmat,chord = batch
                multi_prmat = multi_prmat.to(self.device)
                chord = chord.to(self.device)
                with self.autocast:
                    loss = self.diffusion.loss(multi_prmat, chord)

                running_loss.append(loss.item())

                loss.backward()
                self.optimizer.step()

                if self.step % self.log_train_loss_interval == 0:
                    self.writer.add_scalar('Training Loss', np.mean(running_loss), self.step)                 

                    running_loss = []
                    self.writer.flush()
                
                if self.step % self.validation_interval == 0 and self.step != 0:
                    self.diffusion.eval()
                    val_loss = []
                    with torch.no_grad():
                        
                        for batch in tqdm(self.val_loader):
                            multi_prmat,chord = batch
                            multi_prmat = multi_prmat.to(self.device)
                            chord = chord.to(self.device)
                            loss = self.diffusion.loss(multi_prmat, chord)
                            val_loss.append(loss.item())
                    val_loss = np.mean(val_loss)
                    self.writer.add_scalar('Validation Loss', val_loss, self.step)
                    self.writer.flush()
                    self.save_checkpoint(False)
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint(True)                

                self.step += 1
            self.epoch += 1
    
    def save_checkpoint(self,is_best = False):
        fpath = "best.pth" if is_best else "last.pth"
        checkpoint_path = os.path.join(self.ckpt_dir, fpath)
        torch.save({
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': self.diffusion.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.diffusion.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch'] + 1
        self.step = checkpoint['step'] + 1
        print(f"Checkpoint loaded from {checkpoint_path}, resuming from epoch {self.epoch} and step {self.step}")
      



    
