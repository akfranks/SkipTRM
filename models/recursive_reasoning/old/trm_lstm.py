"""
LSTM-based Tiny Recursive Model (LSTM-TRM)

This variant replaces the recurrent loop in TRM with an LSTM where:
- Cell state (c) corresponds to y in TRM (updated every T steps - once per full recursion)
- Hidden state (h) corresponds to z in TRM (updated every n steps - every timestep)
- Output uses cell state c, not hidden state h
- Backpropagation only on final cell state update
"""

from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding

IGNORE_LABEL_ID = -100


@dataclass
class LSTMTRMInnerCarry:
    """Carry state for LSTM-TRM containing cell and hidden states"""
    c: torch.Tensor  # Cell state (like y/z_H in TRM)
    h: torch.Tensor  # Hidden state (like z/z_L in TRM)


@dataclass
class LSTMTRMCarry:
    """Outer carry with halting mechanism"""
    inner_carry: LSTMTRMInnerCarry

    steps: torch.Tensor
    halted: torch.Tensor

    current_data: Dict[str, torch.Tensor]


class LSTMTRMConfig(BaseModel):
    """Configuration for LSTM-based TRM"""
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    # Recursion parameters
    T_cycles: int  # Number of full recursions (like H_cycles in TRM)
    n_cycles: int  # Number of h updates per recursion (like L_cycles in TRM)

    # Network config
    hidden_size: int
    expansion: float = 2.0

    rms_norm_eps: float = 1e-5

    # Halting config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"

    # Additional options
    puzzle_emb_len: int = 16
    no_ACT_continue: bool = True


class LSTMCell(nn.Module):
    """
    Custom LSTM cell for TRM.

    This LSTM separates the update frequencies:
    - Hidden state h is updated every step (n times)
    - Cell state c is updated once per n steps
    """
    def __init__(self, config: LSTMTRMConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        # LSTM gates for updating h (hidden state)
        # Input: [x, c, h] concatenated
        self.h_gate = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion
        )

        # Gate for updating c (cell state)
        # Input: [c, h] concatenated
        self.c_gate = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion
        )

        self.norm_eps = config.rms_norm_eps

    def update_h(self, x: torch.Tensor, c: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Update hidden state h given input x, cell state c, and previous h.
        Cell state c remains unchanged.
        """
        # Combine inputs: x + c + h (similar to z_L = net(z_L, z_H + x))
        combined = x + c + h
        combined = rms_norm(combined, variance_epsilon=self.norm_eps)

        # Update h
        h_new = self.h_gate(combined)
        h_out = rms_norm(h + h_new, variance_epsilon=self.norm_eps)

        return h_out

    def update_c(self, c: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Update cell state c given final hidden state h.
        Similar to y = net(y, z) in TRM.
        """
        # Combine c and h (similar to z_H = net(z_H, z_L))
        combined = c + h
        combined = rms_norm(combined, variance_epsilon=self.norm_eps)

        # Update c
        c_new = self.c_gate(combined)
        c_out = rms_norm(c + c_new, variance_epsilon=self.norm_eps)

        return c_out


class LSTMTRMInner(nn.Module):
    """Inner LSTM-TRM model with recursion logic"""

    def __init__(self, config: LSTMTRMConfig):
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # Embeddings
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            init_std=embed_init_std,
            cast_to=self.forward_dtype
        )

        # Output heads
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        # Puzzle embeddings
        self.puzzle_emb_len = (
            -(self.config.puzzle_emb_ndim // -self.config.hidden_size)
            if self.config.puzzle_emb_len == 0
            else self.config.puzzle_emb_len
        )

        if self.config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                self.config.num_puzzle_identifiers,
                self.config.puzzle_emb_ndim,
                batch_size=self.config.batch_size,
                init_std=0,
                cast_to=self.forward_dtype
            )

        # LSTM cell
        self.lstm_cell = LSTMCell(config)

        # Initial states
        self.c_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True
        )
        self.h_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True
        )

        # Q head initialization
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        """Compute input embeddings with optional puzzle embeddings"""
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)

            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat(
                (puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding),
                dim=-2
            )

        # Scale
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int, device=None):
        """Create empty carry state"""
        if device is None:
            device = self.c_init.device
        return LSTMTRMInnerCarry(
            c=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype, device=device),
            h=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype, device=device),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: LSTMTRMInnerCarry):
        """Reset carry state for halted sequences"""
        return LSTMTRMInnerCarry(
            c=torch.where(reset_flag.view(-1, 1, 1), self.c_init.to(carry.c.device), carry.c),
            h=torch.where(reset_flag.view(-1, 1, 1), self.h_init.to(carry.h.device), carry.h),
        )

    def forward(
        self,
        carry: LSTMTRMInnerCarry,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[LSTMTRMInnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with LSTM recursion.

        Structure:
        - T-1 recursions without gradients:
          - n updates to h (with c fixed)
          - 1 update to c (based on final h)
        - 1 recursion with gradients:
          - n updates to h
          - 1 update to c
        """
        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        c, h = carry.c, carry.h

        # T-1 recursions without gradients
        with torch.no_grad():
            for _T_step in range(self.config.T_cycles - 1):
                # Update h for n steps (with c fixed)
                for _n_step in range(self.config.n_cycles):
                    h = self.lstm_cell.update_h(input_embeddings, c, h)
                # Update c once (based on final h)
                c = self.lstm_cell.update_c(c, h)

        # 1 recursion with gradients
        for _n_step in range(self.config.n_cycles):
            h = self.lstm_cell.update_h(input_embeddings, c, h)
        c = self.lstm_cell.update_c(c, h)

        # Output from cell state c (not hidden state h!)
        new_carry = LSTMTRMInnerCarry(c=c.detach(), h=h.detach())
        output = self.lm_head(c)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(c[:, 0]).to(torch.float32)

        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class LSTMTRM(nn.Module):
    """LSTM-based Tiny Recursive Model with ACT wrapper"""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = LSTMTRMConfig(**config_dict)
        self.inner = LSTMTRMInner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        """Initialize carry state"""
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device

        return LSTMTRMCarry(
            inner_carry=self.inner.empty_carry(batch_size, device=device),
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=device),
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )

    def forward(
        self,
        carry: LSTMTRMCarry,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[LSTMTRMCarry, Dict[str, torch.Tensor]]:
        """
        Forward pass with deep supervision and ACT halting.

        This is called repeatedly (up to halt_max_steps times) for deep supervision.
        """
        # Reset carry for halted sequences
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k],
                v
            )
            for k, v in carry.current_data.items()
        }

        # Forward through inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(
            new_inner_carry,
            new_current_data
        )

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }

        # Halting logic
        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            halted = is_last_step

            if self.training and (self.config.halt_max_steps > 1):
                # Halt when model is confident
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (
                    (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob)
                    * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                )
                halted = halted & (new_steps >= min_halt_steps)

                if not self.config.no_ACT_continue:
                    # Target Q for continue loss
                    _, _, (next_q_halt_logits, next_q_continue_logits) = self.inner(
                        new_inner_carry,
                        new_current_data
                    )
                    outputs["target_q_continue"] = torch.sigmoid(
                        torch.where(
                            is_last_step,
                            next_q_halt_logits,
                            torch.maximum(next_q_halt_logits, next_q_continue_logits)
                        )
                    )

        return LSTMTRMCarry(new_inner_carry, new_steps, halted, new_current_data), outputs
