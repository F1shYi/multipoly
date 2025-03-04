from models.prepare_model import get_diffusion
from data.dataloader import get_train_val_dataloaders
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
import yaml

class Learner:
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.diffusion = get_diffusion(
            config["paths"]["polyffusion"],
            config["paths"]["chord_encoder"],
            config["models"]["transformers"],
            (config["init"]["intratrack"] == "polyffusion"),
            config["training"]["freeze_polyffusion"],
            (config["init"]["intertrack"] == "zero"),
            ).to(self.device)
        
        self.train_loader, self.val_loader = get_train_val_dataloaders(
            config["paths"]["dataset"],
            config["data"]["batch_size"],
            config["data"]["num_workers"],
            config["data"]["train_ratio"],
            pin_memory=True)
        
        self.optimizer = optim.Adam(self.diffusion.parameters(), lr=config["training"]["lr"])



        self.output_dir = config["paths"]["output"]
        self.log_dir = self.output_dir + "/logs"
        self.ckpt_dir = self.output_dir + "/ckpts"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        with open(self.output_dir+"/config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)


        self.autocast = torch.cuda.amp.autocast(enabled=True)

        self.writer = SummaryWriter(self.log_dir)
        self.epoch = 0
        self.step = 0
        self.accumulation_steps = config["training"]["accumulation_steps"]
        self.log_train_loss_interval = 200
        self.validation_interval = 5000
        self.best_val_loss = 1e10
        
        
    def train(self):
        while True:
            running_loss = []
            from tqdm import tqdm
            for batch in tqdm(self.train_loader):
                multi_prmat,chord = batch
                multi_prmat = multi_prmat.to(self.device)
                chord = chord.to(self.device)
                with self.autocast:
                    loss = self.diffusion.loss(multi_prmat, chord)
                running_loss.append(loss.item())
                loss = loss/self.accumulation_steps
                loss.backward()

                if (self.step + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                
                if (self.step + 1) % self.log_train_loss_interval == 0:
                    self.writer.add_scalar('Training Loss', np.mean(running_loss), self.step)
                    running_loss = []
                    self.writer.flush()
                
                if (self.step + 1) % self.validation_interval == 0:
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
                    self.diffusion.train()       
                self.step += 1

            if (self.step + 1) % self.accumulation_steps != 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

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
      



    
