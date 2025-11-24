"""
LSTM Model for Recursive Reasoning

This implements an LSTM-based architecture for puzzle solving with Adaptive Computation Time (ACT).
The model uses stacked LSTM layers to process sequences iteratively, with Q-learning based halting.
"""

from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding


@dataclass
class Model_LSTMInnerCarry:
    """Carry for LSTM hidden and cell states."""
    h: torch.Tensor  # Hidden state
    c: torch.Tensor  # Cell state


@dataclass
class Model_LSTMCarry:
    """Outer carry for ACT wrapper."""
    inner_carry: Model_LSTMInnerCarry

    steps: torch.Tensor
    halted: torch.Tensor

    current_data: Dict[str, torch.Tensor]


class Model_LSTMConfig(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    # LSTM config
    hidden_size: int
    num_layers: int
    dropout: float = 0.0
    bidirectional: bool = False

    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float
    act_enabled: bool = True  # If False, always run halt_max_steps (no early stopping during training)
    act_inference: bool = False  # If True, use adaptive computation during inference

    forward_dtype: str = "bfloat16"

    # Puzzle embedding configuration
    puzzle_emb_len: int = 0  # If non-zero, use this value instead of computing from puzzle_emb_ndim


class Model_LSTM_Inner(nn.Module):
    def __init__(self, config: Model_LSTMConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O embeddings
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            init_std=embed_init_std,
            cast_to=self.forward_dtype,
        )
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            if self.config.puzzle_emb_len == 0:
                self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  # ceil div
            else:
                self.puzzle_emb_len = self.config.puzzle_emb_len

            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(
                self.config.num_puzzle_identifiers,
                self.config.puzzle_emb_ndim,
                batch_size=self.config.batch_size,
                init_std=0,
                cast_to=self.forward_dtype,
            )
        else:
            self.puzzle_emb_len = 0

        # LSTM layers
        # Note: PyTorch LSTM expects input of shape (batch, seq, feature) when batch_first=True
        self.lstm = nn.LSTM(
            input_size=self.config.hidden_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            batch_first=True,
            dropout=self.config.dropout if self.config.num_layers > 1 else 0.0,
            bidirectional=self.config.bidirectional,
        )

        # If bidirectional, we need a projection to reduce back to hidden_size
        lstm_output_size = self.config.hidden_size * (2 if self.config.bidirectional else 1)
        if self.config.bidirectional:
            self.output_proj = CastedLinear(lstm_output_size, self.config.hidden_size, bias=False)

        # Initial states
        num_directions = 2 if self.config.bidirectional else 1
        self.h_init = nn.Buffer(
            trunc_normal_init_(
                torch.empty(
                    self.config.num_layers * num_directions,
                    self.config.hidden_size,
                    dtype=self.forward_dtype
                ),
                std=1
            ),
            persistent=True,
        )
        self.c_init = nn.Buffer(
            trunc_normal_init_(
                torch.empty(
                    self.config.num_layers * num_directions,
                    self.config.hidden_size,
                    dtype=self.forward_dtype
                ),
                std=1
            ),
            persistent=True,
        )

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
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

    def empty_carry(self, batch_size: int):
        num_directions = 2 if self.config.bidirectional else 1
        return Model_LSTMInnerCarry(
            h=torch.empty(
                batch_size,
                self.config.num_layers * num_directions,
                self.config.hidden_size,
                dtype=self.forward_dtype,
            ),
            c=torch.empty(
                batch_size,
                self.config.num_layers * num_directions,
                self.config.hidden_size,
                dtype=self.forward_dtype,
            ),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: Model_LSTMInnerCarry):
        return Model_LSTMInnerCarry(
            h=torch.where(reset_flag.view(-1, 1, 1), self.h_init, carry.h),
            c=torch.where(reset_flag.view(-1, 1, 1), self.c_init, carry.c),
        )

    def forward(
        self, carry: Model_LSTMInnerCarry, batch: Dict[str, torch.Tensor]
    ) -> Tuple[Model_LSTMInnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # LSTM forward
        # PyTorch LSTM expects (h, c) in shape (num_layers * num_directions, batch, hidden_size)
        # Our carry has shape (batch, num_layers * num_directions, hidden_size)
        h_in = carry.h.transpose(0, 1).contiguous()
        c_in = carry.c.transpose(0, 1).contiguous()

        # Convert to float32 for LSTM (PyTorch LSTM only supports float32)
        input_embeddings_f32 = input_embeddings.to(torch.float32)
        h_in_f32 = h_in.to(torch.float32)
        c_in_f32 = c_in.to(torch.float32)

        lstm_out, (h_out, c_out) = self.lstm(input_embeddings_f32, (h_in_f32, c_in_f32))

        # Convert back to forward_dtype
        lstm_out = lstm_out.to(self.forward_dtype)
        h_out = h_out.to(self.forward_dtype)
        c_out = c_out.to(self.forward_dtype)

        # If bidirectional, project back to hidden_size
        if self.config.bidirectional:
            lstm_out = self.output_proj(lstm_out)

        # Transpose back to (batch, num_layers * num_directions, hidden_size)
        h_out = h_out.transpose(0, 1)
        c_out = c_out.transpose(0, 1)

        # New carry (detached)
        new_carry = Model_LSTMInnerCarry(
            h=h_out.detach(),
            c=c_out.detach(),
        )

        # LM head output (skip puzzle embedding positions)
        output = self.lm_head(lstm_out)[:, self.puzzle_emb_len:]

        # Q head (use first position, typically the puzzle embedding)
        q_logits = self.q_head(lstm_out[:, 0]).to(torch.float32)

        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class Model_LSTM(nn.Module):
    """LSTM model with ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = Model_LSTMConfig(**config_dict)
        self.inner = Model_LSTM_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        return Model_LSTMCarry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),  # Default to halted
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(
        self,
        carry: Model_LSTMCarry,
        batch: Dict[str, torch.Tensor],
        compute_target_q: bool = False,
    ) -> Tuple[Model_LSTMCarry, Dict[str, torch.Tensor]]:
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)

        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {
            k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v)
            for k, v in carry.current_data.items()
        }

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(
            new_inner_carry, new_current_data
        )

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps

            halted = is_last_step

            # Check if adaptive computation should be used
            use_adaptive = (self.config.halt_max_steps > 1) and (
                (self.training and self.config.act_enabled)
                or (not self.training and self.config.act_inference)
            )

            if use_adaptive:
                # Halt signal based on Q-values (but always halt at max steps)
                q_halt_signal = q_halt_logits > q_continue_logits
                halted = halted | q_halt_signal

                # Store actual steps used for logging (only during inference)
                if not self.training:
                    outputs["actual_steps"] = new_steps.float()

                # Exploration (only during training)
                if self.training:
                    min_halt_steps = (
                        torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob
                    ) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                    halted = halted & (new_steps >= min_halt_steps)

                # Compute target Q (only during training)
                if self.training and compute_target_q:
                    next_q_halt_logits, next_q_continue_logits = self.inner(
                        new_inner_carry, new_current_data
                    )[-1]

                    outputs["target_q_continue"] = torch.sigmoid(
                        torch.where(
                            is_last_step,
                            next_q_halt_logits,
                            torch.maximum(next_q_halt_logits, next_q_continue_logits),
                        )
                    )

        return Model_LSTMCarry(
            new_inner_carry, new_steps, halted, new_current_data
        ), outputs
