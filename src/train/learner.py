from models.prepare_model import get_diffusion
from data.dataloader import get_train_val_dataloaders
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.tensorboard import SummaryWriter

POLYFFUSION_CKPT_PATH = "/root/autodl-tmp/multipoly/polyffusion_ckpts/ldm_chd8bar/sdf+pop909wm_mix16_chd8bar/01-11_102022/chkpts/weights_best.pt"
CHORD_ENCODER_CKPT_PATH = "/root/autodl-tmp/multipoly/pretrained/chd8bar/weights.pt"
TRAINABLE_DICT = {"n_intertrack_head":4, "num_intertrack_encoder_layers":1,"intertrack_attention_levels":[2]}
DATAFOLDER = "/root/autodl-tmp/multipoly/data/lmd/lpd_5_midi/"   
BATCH_SIZE = 4
NUM_WORKERS = 4                         
LR = 1e-5

class TorchLearner:
    def __init__(self, output_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.diffusion = get_diffusion(POLYFFUSION_CKPT_PATH, CHORD_ENCODER_CKPT_PATH, TRAINABLE_DICT).to(self.device)
        self.train_loader, self.val_loader = get_train_val_dataloaders(DATAFOLDER, BATCH_SIZE, NUM_WORKERS, pin_memory=True)
        
        self.optimizer = optim.Adam(self.diffusion.parameters(), lr=LR)
        

        self.output_dir = output_dir
        self.log_dir = self.output_dir + "/logs"
        self.ckpt_dir = self.output_dir + "/ckpts"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        
        self.writer = SummaryWriter(self.log_dir)
        self.epoch = 0
        self.step = 0
        
        
    
    def train_step(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy
    
    def val_epoch(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy
    
    def train(self, epochs):
        for epoch in range(self.start_epoch, epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.val_epoch()
            
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/Val', val_acc, epoch)
            
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}% | Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")
            
            self.save_checkpoint(epoch)
    
    def save_checkpoint(self, epoch):
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        print(f"Checkpoint loaded from {checkpoint_path}, resuming from epoch {self.start_epoch}")
      



    
