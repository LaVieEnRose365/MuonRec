import logging
import numpy as np
import torch
import os
import sys
import heapq
from time import time
from torch import optim
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

# =====================================================
# Dynamically add path to ensure internal imports work
# =====================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    # Attempt local import for the engine structure
    from utils import ensure_dir, set_color, get_local_time, delete_file
except ImportError:
    # Fallback if invoked differently
    from minionerec_engine.rq.utils import ensure_dir, set_color, get_local_time, delete_file

class Trainer(object):
    """
    Standard Trainer for RQ-VAE models within the MuonRec framework.
    Handles optimization, scheduling, evaluation, and checkpoint management.
    """

    def __init__(self, args, model, data_num):
        self.args = args
        self.model = model
        self.logger = logging.getLogger()

        self.lr = args.lr
        self.learner = args.learner
        self.lr_scheduler_type = args.lr_scheduler_type
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.warmup_steps = args.warmup_epochs * data_num
        self.max_steps = args.epochs * data_num

        self.save_limit = args.save_limit
        self.best_save_heap = []
        self.newest_save_queue = []
        self.eval_step = min(args.eval_step, self.epochs)
        self.device = torch.device(args.device)
        
        # Setup Checkpoint Directory
        # We save models in a timestamped folder to prevent overwriting previous experiments
        self.base_ckpt_dir = args.ckpt_dir
        timestamp = get_local_time()
        self.ckpt_dir = os.path.join(self.base_ckpt_dir, timestamp)
        ensure_dir(self.ckpt_dir)

        self.best_loss = np.inf
        self.best_collision_rate = np.inf
        self.best_loss_ckpt = "best_loss_model.pth"
        self.best_collision_ckpt = "best_collision_model.pth"
        
        self.optimizer = self._build_optimizer()
        self.scheduler = self._get_scheduler()
        self.model = self.model.to(self.device)

    def _build_optimizer(self):
        """Builds optimizer based on the specified learner type."""
        params = self.model.parameters()
        learner = self.learner.lower()
        
        if learner == "adam":
            return optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        elif learner == "adamw":
            return optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        elif learner == "sgd":
            return optim.SGD(params, lr=self.lr, weight_decay=self.weight_decay)
        elif learner == "adagrad":
            return optim.Adagrad(params, lr=self.lr, weight_decay=self.weight_decay)
        elif learner == "rmsprop":
            return optim.RMSprop(params, lr=self.lr, weight_decay=self.weight_decay)
        else:
            self.logger.warning(f"Unrecognized optimizer '{self.learner}', defaulting to AdamW")
            return optim.AdamW(params, lr=self.lr)

    def _get_scheduler(self):
        """Returns a learning rate scheduler with warmup."""
        if self.lr_scheduler_type.lower() == "linear":
            return get_linear_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.max_steps
            )
        return get_constant_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.warmup_steps
        )

    def _train_epoch(self, train_data, epoch_idx):
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        
        iter_data = tqdm(
            train_data,
            total=len(train_data),
            ncols=100,
            desc=set_color(f"Train Epoch {epoch_idx}", "pink"),
        )

        for batch_idx, data in enumerate(iter_data):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            
            out, rq_loss, indices = self.model(data)
            loss, loss_recon = self.model.compute_loss(out, rq_loss, xs=data)
            
            if torch.isnan(loss):
                raise ValueError("Training loss reached NaN")
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            total_recon_loss += loss_recon.item()

        return total_loss, total_recon_loss

    @torch.no_grad()
    def _valid_epoch(self, valid_data):
        self.model.eval()
        indices_set = set()
        num_sample = 0
        
        iter_data = tqdm(
            valid_data,
            total=len(valid_data),
            ncols=100,
            desc=set_color("Evaluating", "blue"),
        )

        for data in iter_data:
            num_sample += len(data)
            data = data.to(self.device)
            indices = self.model.get_indices(data)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for index in indices:
                code = "-".join([str(int(_)) for _ in index])
                indices_set.add(code)

        collision_rate = (num_sample - len(indices_set)) / num_sample
        return collision_rate

    def _save_checkpoint(self, epoch, collision_rate=1.0, ckpt_file=None, is_best=False):
        """Saves a model checkpoint. If is_best is True, saves to the base directory for easy access."""
        # Best models are saved in the root of ckpt_dir for easier retrieval by indexing scripts
        save_dir = self.base_ckpt_dir if is_best else self.ckpt_dir
        ckpt_path = os.path.join(save_dir, ckpt_file) if ckpt_file \
            else os.path.join(self.ckpt_dir, f'epoch_{epoch}_collision_{collision_rate:.4f}.pth')
            
        state = {
            "args": self.args,
            "epoch": epoch,
            "best_loss": self.best_loss,
            "best_collision_rate": self.best_collision_rate,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, ckpt_path, pickle_protocol=4)
        return ckpt_path

    def fit(self, data):
        """Standard training loop."""
        for epoch_idx in range(self.epochs):
            training_start_time = time()
            train_loss, train_recon_loss = self._train_epoch(data, epoch_idx)
            training_end_time = time()
            
            self.logger.info(
                f"Epoch {epoch_idx} training completed in {training_end_time-training_start_time:.2f}s | "
                f"Loss: {train_loss:.4f} | Recon Loss: {train_recon_loss:.4f}"
            )

            # Periodic Evaluation
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                collision_rate = self._valid_epoch(data)
                valid_end_time = time()

                # Save if best Loss
                if train_loss < self.best_loss:
                    self.best_loss = train_loss
                    self._save_checkpoint(epoch=epoch_idx, ckpt_file=self.best_loss_ckpt, is_best=True)

                # Save if best Collision Rate
                if collision_rate < self.best_collision_rate:
                    self.best_collision_rate = collision_rate
                    ckpt_path = self._save_checkpoint(epoch_idx, collision_rate=collision_rate, 
                                                      ckpt_file=self.best_collision_ckpt, is_best=True)
                    self.logger.info(set_color(f"New Best Collision Rate: {collision_rate:.6f}", "red"))

                self.logger.info(
                    f"Epoch {epoch_idx} evaluating completed in {valid_end_time-valid_start_time:.2f}s | "
                    f"Collision Rate: {collision_rate:.6f}"
                )
                
                # Save regular checkpoint and manage disk space
                ckpt_path = self._save_checkpoint(epoch_idx, collision_rate=collision_rate)
                now_save = (-collision_rate, ckpt_path)
                
                if len(self.newest_save_queue) < self.save_limit:
                    self.newest_save_queue.append(now_save)
                    heapq.heappush(self.best_save_heap, now_save)
                else:
                    old_save = self.newest_save_queue.pop(0)
                    self.newest_save_queue.append(now_save)
                    if collision_rate < -self.best_save_heap[0][0]:
                        bad_save = heapq.heappop(self.best_save_heap)
                        heapq.heappush(self.best_save_heap, now_save)
                        if bad_save not in self.newest_save_queue:
                            delete_file(bad_save[1])
                    if old_save not in self.best_save_heap:
                        delete_file(old_save[1])

        return self.best_loss, self.best_collision_rate