from models.prepare_model import get_diffusion
from data.dataloader import get_train_val_datas
from data.utils import prmat2c_to_midi_file, chd_to_midi_file, multi_prmat2c_to_midi_file
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import os
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
import yaml

def get_lr_lambda(warmup_steps, decay_factor, decay_interval):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps  # Linear warmup
        return decay_factor ** ((step - warmup_steps) // decay_interval)  # Step decay
    return lr_lambda



class Learner:
    def __init__(self, config):

        self.output_dir = config["paths"]["output"]
        self.log_dir = self.output_dir + "/logs"
        self.ckpt_dir = self.output_dir + "/ckpts"
        self.val_dir = self.output_dir + "/vals"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.val_dir, exist_ok=True)

        with open(self.output_dir+"/config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.diffusion = get_diffusion(
            config["paths"]["polyffusion"],
            config["paths"]["chord_encoder"],
            config["models"]["transformers"],
            (config["init"]["intratrack"] == "polyffusion"),
            config["training"]["freeze_polyffusion"],
            (config["init"]["intertrack"] == "zero"),
            ).to(self.device)
        
        self.train_ds, self.val_ds, self.train_dl, self.val_dl = get_train_val_datas(
            config["paths"]["train_folder"],
            config["paths"]["val_folder"],
            config["data"]["train_bs"],
            config["data"]["val_bs"],
            config["data"]["num_workers"],
            pin_memory=True,
            )
        
       
        # Generate samples at each validation step
        self.val_segs_idx = []
        for _ in range(config["validation"]["num_seg"]):
            while True:
                idx = np.random.randint(0,len(self.val_ds))
                if idx not in self.val_segs_idx:
                    self.val_segs_idx.append(idx)
                    break
        assert len(self.val_segs_idx) == config["validation"]["num_seg"]
        self.val_segs = [self.val_ds[idx] for idx in self.val_segs_idx]
        self.val_num_gen = config["validation"]["num_gen_per_seg"]

        # save chosen validation segments and chords
        NAME = ["bass", "guitar", "piano", "string"]
        for seg_idx, seg in enumerate(self.val_segs):
            prmat, chord = seg
            os.makedirs(os.path.join(self.val_dir,f"seg_{seg_idx}"),exist_ok=True)
            chord_fpath = os.path.join(self.val_dir,f"seg_{seg_idx}", f"val_{seg_idx}_chord.mid")
            multi_fpath = os.path.join(self.val_dir,f"seg_{seg_idx}", f"val_{seg_idx}_multi.mid")
            chd_to_midi_file(chord, chord_fpath)
            multi_prmat2c_to_midi_file(prmat, multi_fpath)
            for track_idx, track in enumerate(prmat):
                midi_fpath = os.path.join(self.val_dir, f"seg_{seg_idx}",f"val_{seg_idx}_track_{NAME[track_idx]}.mid")
                prmat2c_to_midi_file(track, midi_fpath)



        self.optimizer = optim.Adam(self.diffusion.parameters(), lr=config["training"]["lr"])
        self.scheduler = LambdaLR(self.optimizer, 
                                  lr_lambda=get_lr_lambda(config["training"]["warmup_steps"],
                                                        config["training"]["decay_factor"],
                                                        config["training"]["decay_interval"])
                                )



        self.writer = SummaryWriter(self.log_dir)
        self.epoch = 0
        self.step = 0
        
        self.accumulation_steps = config["training"]["accumulation_steps"]
        self.log_train_loss_interval = config["training"]["log_train_loss_interval"]
        self.validation_interval = config["training"]["validation_interval"]
        self.generate_chord_conditioned_samples_interval = config["training"]["generate_chord_conditioned_samples_interval"]
        self.generate_track_given_samples_interval = config["training"]["generate_track_given_samples_interval"]
        
        self.best_val_loss = 1e10
        
    

        

    def train(self):
        while True:
            running_loss = []
            from tqdm import tqdm
            for batch in tqdm(self.train_dl):
                multi_prmat,chord = batch
                multi_prmat = multi_prmat.to(self.device)
                chord = chord.to(self.device)
                loss = self.diffusion.loss(multi_prmat, chord)
                running_loss.append(loss.item())
                loss = loss/self.accumulation_steps
                loss.backward()

                if (self.step + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    current_lr = self.scheduler.get_last_lr()[0]
                    self.writer.add_scalar("Learning Rate", current_lr, self.step)

                if (self.step + 1) % self.log_train_loss_interval == 0:
                    self.writer.add_scalar('Training Loss', np.mean(running_loss), self.step)
                    running_loss = []
                    self.writer.flush()
                
                if (self.step + 1) % self.validation_interval == 0:
                    self.validation_step()
                    self.diffusion.train()

                if (self.step + 1) % self.generate_chord_conditioned_samples_interval == 0:
                    self.generate_chord_conditioned_samples()
                    self.diffusion.train()

                if (self.step + 1) % self.generate_track_given_samples_interval == 0:
                    self.generate_track_given_samples()
                    self.diffusion.train()

                self.step += 1

            if (self.step + 1) % self.accumulation_steps != 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            self.epoch += 1
    
    @torch.no_grad()
    def validation_step(self):
        from tqdm import tqdm
        self.diffusion.eval()
        val_loss = []
        with torch.no_grad():
            for batch in tqdm(self.val_dl):
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

    @torch.no_grad()
    def generate_chord_conditioned_samples(self):
        self.diffusion.eval()
        NAME = ["bass", "guitar", "piano", "string"]
        for seg_idx, seg in enumerate(self.val_segs):        
            multi_prmat, chord = seg
            
            save_folder = os.path.join(self.val_dir, f"seg_{seg_idx}", f"step_{self.step}")
            os.makedirs(save_folder, exist_ok=True)
            multi_prmat = torch.from_numpy(multi_prmat).to(self.device).reshape(1,*multi_prmat.shape)
            chord = torch.from_numpy(chord).to(self.device).reshape(1, *chord.shape)
            
            for gen_idx in range(self.val_num_gen):
                print(f"generating song {gen_idx} for validation segment {seg_idx} at step {self.step}")

                for uncond_scale in [0.5,2.0,5.0]:

                    gen_folder = os.path.join(save_folder, f"scale_{uncond_scale}",f"gen_{gen_idx}")
                    os.makedirs(gen_folder, exist_ok=True)

                    gen = self.diffusion.sample(
                    shape=[1,4,2,128,128],
                    chords=chord,
                    uncond_scale=uncond_scale,
                    uncond_cond=-(torch.ones([1,1,512])).to(self.device))
                    gen = gen.detach().cpu().numpy()[0]

                    multi_prmat2c_to_midi_file(gen, os.path.join(gen_folder,f"multi.mid"))
                    for track_idx, track in enumerate(gen):
                        midi_fpath = os.path.join(gen_folder, f"track_{NAME[track_idx]}.mid")
                        prmat2c_to_midi_file(track, midi_fpath)

           
    
    @torch.no_grad()
    def generate_track_given_samples(self):
        # TODO
        pass
       


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