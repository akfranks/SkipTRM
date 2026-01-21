"""
Refactored Training Script for TRM/HRM/Skip-TRM Models
=======================================================
Standard PyTorch training organization while preserving unique TRM features:
- Deep supervision (handled inside model's forward pass)
- ACT halting (Adaptive Computation Time)
- Carry state (maintained across batches)
- Distributed training support

Usage:
    # Import and use the Trainer class
    from train_refactored import Trainer, TrainingConfig

    config = TrainingConfig(...)
    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.fit()

    # Or run this file directly for a quick test
    python train_refactored.py test
"""

import os
import math
from typing import Optional, Any, Dict, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Model & data
    model_config: Dict[str, Any]
    vocab_size: int
    seq_len: int

    # Training hyperparameters
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    beta1: float = 0.9
    beta2: float = 0.95
    lr_warmup_steps: int = 2000
    lr_min_ratio: float = 0.1

    # Checkpointing & logging
    checkpoint_dir: str = "./checkpoints"
    log_interval: int = 10
    eval_interval: int = 1000
    save_interval: int = 1000

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.999

    # Distributed training
    distributed: bool = False
    world_size: int = 1
    rank: int = 0


class Trainer:
    """
    Standard PyTorch trainer for TRM/HRM/Skip-TRM models.

    Handles the unique aspects of recursive reasoning models:
    - Carry state management across batches
    - ACT halting logic (via model forward pass)
    - Deep supervision (via model forward pass)
    - EMA model averaging
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: TrainingConfig,
        device: str = "cuda"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Move model to device
        self.model.to(device)

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Setup learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Setup EMA if requested
        self.ema_model = None
        if config.use_ema:
            self.ema_model = self._create_ema_model()

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.carry = None  # For TRM/HRM carry state

        # Metrics tracking
        self.train_metrics = {}
        self.val_metrics = {}

    def _create_optimizer(self) -> Optimizer:
        """Create optimizer."""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            betas=(self.config.beta1, self.config.beta2)
        )

    def _create_scheduler(self) -> LambdaLR:
        """Create learning rate scheduler with warmup and cosine decay."""
        def lr_lambda(step):
            if step < self.config.lr_warmup_steps:
                return step / max(1, self.config.lr_warmup_steps)

            progress = (step - self.config.lr_warmup_steps) / max(
                1,
                len(self.train_loader) * self.config.epochs - self.config.lr_warmup_steps
            )
            return self.config.lr_min_ratio + 0.5 * (1 - self.config.lr_min_ratio) * (
                1 + math.cos(math.pi * progress)
            )

        return LambdaLR(self.optimizer, lr_lambda)

    def _create_ema_model(self) -> nn.Module:
        """Create EMA copy of the model."""
        import copy
        ema_model = copy.deepcopy(self.model)
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad = False
        return ema_model

    def _update_ema(self):
        """Update EMA model parameters."""
        if self.ema_model is None:
            return

        with torch.no_grad():
            for ema_param, param in zip(
                self.ema_model.parameters(),
                self.model.parameters()
            ):
                ema_param.data.mul_(self.config.ema_decay).add_(
                    param.data,
                    alpha=1 - self.config.ema_decay
                )

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = {
            'loss': 0.0,
            'lm_loss': 0.0,
            'q_halt_loss': 0.0,
            'accuracy': 0.0,
            'exact_accuracy': 0.0,
            'steps': 0.0,
            'count': 0
        }

        pbar = tqdm.tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, batch_data in enumerate(pbar):
            # Move batch to device
            batch = self._prepare_batch(batch_data)

            # Train one step
            metrics = self.train_step(batch)

            # Accumulate metrics
            for key in epoch_metrics:
                if key in metrics:
                    epoch_metrics[key] += metrics[key]

            # Update progress bar
            if self.global_step % self.config.log_interval == 0:
                avg_loss = epoch_metrics['loss'] / max(epoch_metrics['count'], 1)
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})

            self.global_step += 1

        # Normalize epoch metrics
        count = max(epoch_metrics.pop('count'), 1)
        epoch_metrics = {k: v / count for k, v in epoch_metrics.items()}

        return epoch_metrics

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step.

        Handles TRM/HRM specific training logic:
        1. Initialize or reuse carry state
        2. Forward pass (includes deep supervision internally)
        3. Backward pass
        4. Optimizer step
        5. EMA update
        """
        # Initialize carry if needed (TRM/HRM specific)
        if self.carry is None:
            with torch.no_grad():
                self.carry = self.model.initial_carry(batch)

        # Forward pass
        # The model's forward pass handles deep supervision internally
        # Returns: new_carry, loss, metrics, outputs, halted_flag
        new_carry, loss, metrics, outputs, all_halted = self.model(
            carry=self.carry,
            batch=batch,
            return_keys=[]
        )

        # Scale loss by batch size if using gradient accumulation
        if self.config.distributed:
            loss = loss / self.config.world_size

        # Backward pass
        loss.backward()

        # Distributed gradient synchronization
        if self.config.distributed:
            self._sync_gradients()

        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()

        # Update EMA
        if self.config.use_ema:
            self._update_ema()

        # Update carry state for next batch
        self.carry = new_carry

        # Convert metrics to float
        metrics_out = {k: v.item() if torch.is_tensor(v) else v for k, v in metrics.items()}
        metrics_out['loss'] = loss.item()
        metrics_out['count'] = 1

        return metrics_out

    def _prepare_batch(self, batch_data: Any) -> Dict[str, torch.Tensor]:
        """Prepare batch for training."""
        if isinstance(batch_data, (tuple, list)):
            # Handle (set_name, batch, global_batch_size) format
            _, batch, _ = batch_data
        else:
            batch = batch_data

        # Move to device
        if isinstance(batch, dict):
            return {k: v.to(self.device) if torch.is_tensor(v) else v
                   for k, v in batch.items()}
        return batch

    def _sync_gradients(self):
        """Synchronize gradients across distributed processes."""
        if not self.config.distributed:
            return

        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set."""
        if self.val_loader is None:
            return {}

        # Use EMA model for evaluation if available
        model = self.ema_model if self.ema_model is not None else self.model
        model.eval()

        eval_metrics = {
            'loss': 0.0,
            'accuracy': 0.0,
            'exact_accuracy': 0.0,
            'count': 0
        }

        # Reset carry for evaluation
        eval_carry = None

        for batch_data in tqdm.tqdm(self.val_loader, desc="Evaluating"):
            batch = self._prepare_batch(batch_data)

            # Initialize carry if needed
            if eval_carry is None:
                eval_carry = model.initial_carry(batch)

            # Forward pass
            new_carry, loss, metrics, _, _ = model(
                carry=eval_carry,
                batch=batch,
                return_keys=[]
            )

            # Accumulate metrics
            for key in eval_metrics:
                if key in metrics:
                    eval_metrics[key] += metrics[key].item() if torch.is_tensor(metrics[key]) else metrics[key]
            eval_metrics['loss'] += loss.item()
            eval_metrics['count'] += 1

            eval_carry = new_carry

        # Normalize metrics
        count = max(eval_metrics.pop('count'), 1)
        eval_metrics = {k: v / count for k, v in eval_metrics.items()}

        return eval_metrics

    def save_checkpoint(self, path: Optional[str] = None):
        """Save model checkpoint."""
        if path is None:
            path = os.path.join(
                self.config.checkpoint_dir,
                f"checkpoint_step_{self.global_step}.pt"
            )

        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'config': self.config,
        }

        if self.ema_model is not None:
            checkpoint['ema_state_dict'] = self.ema_model.state_dict()

        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['current_epoch']
        self.global_step = checkpoint['global_step']

        if self.ema_model is not None and 'ema_state_dict' in checkpoint:
            self.ema_model.load_state_dict(checkpoint['ema_state_dict'])

        print(f"Loaded checkpoint from {path}")

    def fit(self):
        """Full training loop."""
        print(f"Starting training for {self.config.epochs} epochs")
        print(f"Total steps: {len(self.train_loader) * self.config.epochs}")

        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch

            # Train epoch
            train_metrics = self.train_epoch()

            # Log metrics
            if self.config.rank == 0:
                print(f"\nEpoch {epoch} - Train metrics:")
                for key, value in train_metrics.items():
                    print(f"  {key}: {value:.4f}")

                if WANDB_AVAILABLE and wandb.run is not None:
                    wandb.log({f"train/{k}": v for k, v in train_metrics.items()},
                             step=self.global_step)

            # Evaluate
            if (epoch + 1) % self.config.eval_interval == 0:
                val_metrics = self.evaluate()

                if self.config.rank == 0:
                    print(f"Epoch {epoch} - Val metrics:")
                    for key, value in val_metrics.items():
                        print(f"  {key}: {value:.4f}")

                    if WANDB_AVAILABLE and wandb.run is not None:
                        wandb.log({f"val/{k}": v for k, v in val_metrics.items()},
                                 step=self.global_step)

            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0 and self.config.rank == 0:
                self.save_checkpoint()

        print("Training complete!")


# ==============================================================================
# Quick Test
# ==============================================================================

def quick_test():
    """Quick test to verify the refactored trainer works."""
    print("Running quick test of refactored trainer...")

    # Create dummy model that mimics TRM interface
    class DummyModel(nn.Module):
        def __init__(self, vocab_size=11, hidden_size=64, seq_len=81):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, hidden_size)
            self.linear = nn.Linear(hidden_size, vocab_size)
            self.q_head = nn.Linear(hidden_size, 2)
            self.hidden_size = hidden_size
            self.seq_len = seq_len

        def initial_carry(self, batch):
            B = batch["inputs"].shape[0]
            return {
                "hidden": torch.zeros(B, self.seq_len, self.hidden_size, device=batch["inputs"].device),
                "halted": torch.ones(B, dtype=torch.bool, device=batch["inputs"].device),
                "current_data": batch
            }

        def forward(self, carry, batch, return_keys=None):
            x = self.embed(batch["inputs"])
            logits = self.linear(x)
            q_logits = self.q_head(x[:, 0, :])

            # Dummy loss
            labels = batch["labels"]
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            q_halt_loss = nn.functional.binary_cross_entropy_with_logits(
                q_logits[:, 0],
                torch.zeros_like(q_logits[:, 0])
            )

            metrics = {
                "lm_loss": loss.detach(),
                "q_halt_loss": q_halt_loss.detach(),
                "accuracy": torch.tensor(0.5),
                "exact_accuracy": torch.tensor(0.0),
                "count": torch.tensor(1),
            }

            new_carry = {
                "hidden": x,
                "halted": torch.ones(x.shape[0], dtype=torch.bool, device=x.device),
                "current_data": batch
            }

            return new_carry, loss + 0.5 * q_halt_loss, metrics, {}, True

    # Create dummy dataset
    from torch.utils.data import Dataset, DataLoader

    class DummyDataset(Dataset):
        def __init__(self, size=100, seq_len=81, vocab_size=11):
            self.size = size
            self.seq_len = seq_len
            self.vocab_size = vocab_size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return {
                "inputs": torch.randint(0, self.vocab_size, (self.seq_len,)),
                "labels": torch.randint(0, self.vocab_size, (self.seq_len,)),
            }

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DummyModel()
    train_loader = DataLoader(DummyDataset(size=32), batch_size=8, shuffle=True)
    val_loader = DataLoader(DummyDataset(size=16), batch_size=8)

    config = TrainingConfig(
        model_config={},
        vocab_size=11,
        seq_len=81,
        batch_size=8,
        epochs=2,
        lr=1e-3,
        weight_decay=0.1,
        use_ema=False,
        log_interval=1,
        eval_interval=1,
        save_interval=100,
        checkpoint_dir="./test_checkpoints",
    )

    trainer = Trainer(model, train_loader, val_loader, config, device=device)
    trainer.fit()

    print("\nQuick test passed!")
    return True


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        quick_test()
    else:
        print("Usage: python train_refactored.py test")
        print("\nThis module provides the Trainer class for PyTorch-style training.")
        print("Import and use it in your training scripts.")
