"""
Skip-TRM with Sum-of-Function-Outputs (SOF) Architecture
=========================================================

This is a variant of the Skip-TRM architecture where instead of:
    z_t = f(x + Σ W_i * z_{t-i})         [sum inputs, then call function once]

We use:
    z_t = f(x) + Σ f(W_i * z_{t-i})      [call function for each input, then sum outputs]

This architecture tests whether the model benefits from having the function
process each skip connection independently before aggregation.

Key differences from standard Skip-TRM:
1. Each skip connection is processed through its own L_level pass
2. The input embedding is processed through a separate L_level pass
3. All outputs are summed together
4. This increases computational cost but may improve expressivity

Based on the recommendation from 9.52 report:
"implement a variant that instead of summing skips and inputs into an argument
 used for one function call, calls the function for every skip and input before
 summing the outputs of these calls."
"""

from typing import Tuple, List, Dict
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding

IGNORE_LABEL_ID = -100


@dataclass
class TinyRecursiveReasoningModel_SOF_InnerCarry:
    """Carry state for SOF Skip-TRM."""
    zs: torch.Tensor  # buffer of the last max(skips) hidden states


@dataclass
class TinyRecursiveReasoningModel_SOF_Carry:
    """Full carry state including ACT information."""
    inner_carry: TinyRecursiveReasoningModel_SOF_InnerCarry

    steps: torch.Tensor
    halted: torch.Tensor

    current_data: Dict[str, torch.Tensor]


class TinyRecursiveReasoningModel_SOF_Config(BaseModel):
    """Configuration for SOF Skip-TRM model."""
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int

    skips: List[int]

    H_layers: int  # ignored
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"

    # Architecture options
    mlp_t: bool = False  # use mlp on L instead of transformer
    puzzle_emb_len: int = 16  # if non-zero, its specified to this value
    no_ACT_continue: bool = True  # No continue ACT loss

    # Skip-TRM specific
    output_layers: int = 0  # number of transformer blocks before LM head
    sliding_skips: bool = True  # sliding vs fixed skip connections


class TinyRecursiveReasoningModel_SOF_Block(nn.Module):
    """Transformer/MLP block for SOF architecture."""

    def __init__(self, config: TinyRecursiveReasoningModel_SOF_Config) -> None:
        super().__init__()

        self.config = config
        if self.config.mlp_t:
            self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
            self.mlp_t = SwiGLU(
                hidden_size=self.config.seq_len + self.puzzle_emb_len,
                expansion=config.expansion,
            )
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False
            )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # Post Norm
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1, 2)
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1, 2)
        else:
            hidden_states = rms_norm(
                hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
                variance_epsilon=self.norm_eps
            )
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states


class TinyRecursiveReasoningModel_SOF_ReasoningModule(nn.Module):
    """Reasoning module that processes inputs through transformer layers."""

    def __init__(self, layers: List[TinyRecursiveReasoningModel_SOF_Block]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class TinyRecursiveReasoningModel_SOF_Inner(nn.Module):
    """
    Inner SOF Skip-TRM model.

    Key difference from standard Skip-TRM:
    - Standard: z_t = f(x + Σ W_i * z_{t-i})
    - SOF:      z_t = f(x) + Σ f(W_i * z_{t-i})

    Each skip connection is processed through L_level independently.
    """

    def __init__(self, config: TinyRecursiveReasoningModel_SOF_Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(
            self.config.vocab_size, self.config.hidden_size,
            init_std=embed_init_std, cast_to=self.forward_dtype
        )
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
        if self.config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype
            )

        # Position encodings
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                base=self.config.rope_theta
            )
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size,
                init_std=embed_init_std, cast_to=self.forward_dtype
            )

        # Skip Weights
        self.skips = self.config.skips
        self.skip_weights = torch.nn.ParameterDict({
            f'w_{skip}': torch.nn.Parameter(
                torch.randn(self.config.hidden_size, self.config.hidden_size, dtype=self.forward_dtype)
            )
            for skip in self.skips
        })
        self.max_skip = max(self.skips)

        # Main reasoning layers (for input processing)
        self.L_level_input = TinyRecursiveReasoningModel_SOF_ReasoningModule(
            layers=[TinyRecursiveReasoningModel_SOF_Block(self.config) for _ in range(self.config.L_layers)]
        )

        # Separate reasoning layers for each skip connection
        # This allows each skip to have its own processing pathway
        self.L_level_skips = nn.ModuleDict({
            f'L_{skip}': TinyRecursiveReasoningModel_SOF_ReasoningModule(
                layers=[TinyRecursiveReasoningModel_SOF_Block(self.config) for _ in range(self.config.L_layers)]
            )
            for skip in self.skips
        })

        # Output Layers
        if self.config.output_layers > 0:
            self.output_blocks = torch.nn.ModuleList([
                TinyRecursiveReasoningModel_SOF_Block(self.config)
                for _ in range(self.config.output_layers)
            ])

        # Initial states
        self.L_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True
        )

        # Q head special init
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        """Compute input embeddings with puzzle embeddings and position encodings."""
        embedding = self.embed_tokens(input.to(torch.int32))

        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)

            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat(
                (puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding),
                dim=-2
            )

        if self.config.pos_encodings == "learned":
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        """Create empty carry state initialized with L_init."""
        init_buffer = self.L_init.view(1, 1, -1, 1).expand(
            batch_size, self.config.seq_len + self.puzzle_emb_len,
            self.config.hidden_size, self.max_skip
        ).clone()
        return TinyRecursiveReasoningModel_SOF_InnerCarry(zs=init_buffer)

    def reset_carry(self, reset_flag: torch.Tensor, carry: TinyRecursiveReasoningModel_SOF_InnerCarry):
        """Reset carry state for halted sequences."""
        return TinyRecursiveReasoningModel_SOF_InnerCarry(
            zs=torch.where(reset_flag.view(-1, 1, 1, 1), self.L_init.view(1, 1, -1, 1), carry.zs)
        )

    def forward(
        self,
        carry: TinyRecursiveReasoningModel_SOF_InnerCarry,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[TinyRecursiveReasoningModel_SOF_InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with sum-of-function-outputs architecture.

        Instead of: z_t = f(x + Σ W_i * z_{t-i})
        We use:     z_t = f(x) + Σ f(W_i * z_{t-i})
        """
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Use current buffer for reading
        current_zs = carry.zs
        z = current_zs[..., -1]

        # Zero tensor for input injection (we don't use residual injection in SOF)
        zero_injection = torch.zeros_like(z)

        # H - 1 skip "cycles" without grad (truncate BPTT)
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles - 1):
                for t in range(self.max_skip):
                    # SOF: Process input through L_level
                    z_from_input = self.L_level_input(z, input_embeddings, **seq_info)

                    # SOF: Process each skip connection independently
                    z_sum = z_from_input
                    for skip in self.skips:
                        if self.config.sliding_skips:
                            skip_h = current_zs[..., t - skip]
                        else:
                            skip_h = current_zs[..., (t // skip) * skip]

                        # Apply skip weight and process through skip-specific L_level
                        skip_weighted = torch.matmul(skip_h, self.skip_weights[f'w_{skip}'])
                        z_from_skip = self.L_level_skips[f'L_{skip}'](skip_weighted, zero_injection, **seq_info)
                        z_sum = z_sum + z_from_skip

                    z = z_sum
                    current_zs[..., t] = z  # no grad, so in-place is fine

        # Final H cycle with gradients
        final_zs = [current_zs[..., i].detach().clone() for i in range(self.max_skip)]
        for t in range(self.max_skip):
            # SOF: Process input through L_level
            z_from_input = self.L_level_input(z, input_embeddings, **seq_info)

            # SOF: Process each skip connection independently
            z_sum = z_from_input
            for skip in self.skips:
                if self.config.sliding_skips:
                    skip_z = final_zs[t - skip]
                else:
                    skip_z = final_zs[(t // skip) * skip]

                # Apply skip weight and process through skip-specific L_level
                skip_weighted = torch.matmul(skip_z, self.skip_weights[f'w_{skip}'])
                z_from_skip = self.L_level_skips[f'L_{skip}'](skip_weighted, zero_injection, **seq_info)
                z_sum = z_sum + z_from_skip

            z = z_sum
            final_zs[t] = z

        # LM Outputs
        final_zs = torch.stack(final_zs, dim=-1)  # B, L, D, max_skip
        new_carry = TinyRecursiveReasoningModel_SOF_InnerCarry(zs=final_zs.detach())

        # Get final hidden states for output
        output_hidden = final_zs[..., -1]  # B, L, D

        # Optionally pass through output transformer blocks
        if self.config.output_layers > 0:
            for block in self.output_blocks:
                output_hidden = block(cos_sin=seq_info['cos_sin'], hidden_states=output_hidden)

        # Remove puzzle embeddings and pass to LM head
        output = self.lm_head(output_hidden[:, self.puzzle_emb_len:, :])
        q_logits = self.q_head(final_zs[:, 0, :, -1]).to(torch.float32)
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class TinyRecursiveReasoningModel_ACTV1(nn.Module):
    """
    ACT wrapper for SOF Skip-TRM.

    This is the main entry point for the SOF architecture.
    Uses the same interface as the standard Skip-TRM for compatibility.
    """

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TinyRecursiveReasoningModel_SOF_Config(**config_dict)
        self.inner = TinyRecursiveReasoningModel_SOF_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        return TinyRecursiveReasoningModel_SOF_Carry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )

    def forward(
        self,
        carry: TinyRecursiveReasoningModel_SOF_Carry,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[TinyRecursiveReasoningModel_SOF_Carry, Dict[str, torch.Tensor]]:
        """Forward pass with ACT halting logic."""

        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)

        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k], v
            )
            for k, v in carry.current_data.items()
        }

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

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

            # if training and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1):
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (
                    (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) *
                    torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                )
                halted = halted & (new_steps >= min_halt_steps)

                if not self.config.no_ACT_continue:
                    _, _, (next_q_halt_logits, next_q_continue_logits), _, _ = self.inner(new_inner_carry, new_current_data)
                    outputs["target_q_continue"] = torch.sigmoid(
                        torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits))
                    )

        return TinyRecursiveReasoningModel_SOF_Carry(new_inner_carry, new_steps, halted, new_current_data), outputs
