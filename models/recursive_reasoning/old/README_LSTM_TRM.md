# LSTM-based Tiny Recursive Model (LSTM-TRM)

## Overview

LSTM-TRM is a variant of TRM that replaces the recurrent loop with an LSTM architecture where:

- **Cell state (c)** corresponds to `y` in TRM (the predicted answer)
  - Updated infrequently: once every T cycles
  - This is the "observable state" that gets evaluated for loss

- **Hidden state (h)** corresponds to `z` in TRM (latent reasoning)
  - Updated frequently: n times per recursion
  - This holds intermediate reasoning/memory

## Key Differences from Standard LSTM

Unlike standard LSTMs where both cell state and hidden state are updated at every timestep, LSTM-TRM separates the update frequencies:

1. **For n steps**: Update h (hidden state) while keeping c (cell state) fixed
2. **After n steps**: Update c (cell state) based on the final h
3. **Repeat the above T times**: T-1 times without gradients, 1 time with gradients

## Architecture Mapping

| TRM Component | LSTM-TRM Component | Update Frequency | Role |
|---------------|-------------------|------------------|------|
| y (z_H) | Cell state c | Every T steps (once per full recursion) | Current answer/prediction |
| z (z_L) | Hidden state h | Every step (n times per recursion) | Latent reasoning |
| Output | c (cell state) | - | Final prediction comes from c, not h |

## Implementation Details

### Forward Pass Structure

For each deep supervision step:

```python
# T-1 recursions without gradients
with torch.no_grad():
    for t in range(T-1):
        # Update h n times (c stays fixed)
        for i in range(n):
            h = lstm_cell.update_h(x, c, h)
        # Update c once
        c = lstm_cell.update_c(c, h)

# 1 recursion with gradients
for i in range(n):
    h = lstm_cell.update_h(x, c, h)
c = lstm_cell.update_c(c, h)

# Output prediction from c
output = output_head(c)
```

### Backpropagation

- Only backpropagates through the **final cell state (c) update**
- This matches TRM's approach of backpropagating only through the final y update
- No need for Implicit Function Theorem or fixed-point assumptions

## Configuration

### Parameters

- `T_cycles`: Number of full recursions (like `H_cycles` in TRM)
  - Default: 3
  - Controls how many times we iterate the full recursion process

- `n_cycles`: Number of h updates per recursion (like `L_cycles` in TRM)
  - Default: 6
  - Controls how many times h is updated before updating c

- `hidden_size`: Dimensionality of hidden/cell states
  - Default: 512

- `expansion`: MLP expansion factor in SwiGLU gates
  - Default: 4

- `halt_max_steps`: Maximum number of deep supervision steps
  - Default: 16

### Example Configuration

See `config/arch/trm_lstm.yaml`:

```yaml
name: recursive_reasoning.trm_lstm@LSTMTRM
loss:
  name: losses@ACTLossHead
  loss_type: stablemax_cross_entropy

T_cycles: 3
n_cycles: 6
hidden_size: 512
expansion: 4
halt_max_steps: 16
```

## Usage

### Training

Use the same training script as TRM:

```bash
python pretrain.py arch=trm_lstm [other options]
```

### Model Instantiation

```python
from models.recursive_reasoning.trm_lstm import LSTMTRM, LSTMTRMConfig

config = {
    "batch_size": 64,
    "seq_len": 81,
    "vocab_size": 10,
    "num_puzzle_identifiers": 1000,
    "T_cycles": 3,
    "n_cycles": 6,
    "hidden_size": 512,
    "expansion": 4,
    "halt_max_steps": 16,
    "halt_exploration_prob": 0.1,
    "puzzle_emb_ndim": 512,
    "forward_dtype": "bfloat16"
}

model = LSTMTRM(config)
```

## Advantages over TRM

1. **Clearer separation of concerns**: Cell state c explicitly represents the answer, hidden state h explicitly represents reasoning
2. **No fixed-point assumptions**: No need for Implicit Function Theorem or assumptions about convergence
3. **Simpler interpretation**: LSTM mechanics are well-understood and don't require biological justifications
4. **Parameter efficiency**: Uses same network for both h and c updates (through different gates)

## Comparison with HRM

| Aspect | HRM | TRM | LSTM-TRM |
|--------|-----|-----|----------|
| Networks | 2 (fL, fH) | 1 | 1 (with 2 gates) |
| Layers | 4 each | 2 | 2 (in gates) |
| State variables | 2 (zL, zH) | 2 (z, y) | 2 (h, c) |
| Theoretical justification | Fixed-point + biology | Simplified recursion | LSTM mechanics |
| Backprop | 1-step approx | Full recursion | Full recursion |
| Parameters (typical) | ~27M | ~7M | ~5-7M |

## Expected Performance

Based on TRM performance improvements over HRM, LSTM-TRM should achieve:
- Better or comparable accuracy to TRM
- Parameter efficiency similar to TRM (~5-7M parameters)
- Clearer interpretation of what cell vs hidden states represent

## Notes

- The cell state c serves as memory of the current answer
- The hidden state h performs reasoning to improve the answer
- Gradients flow through the final c update only (like TRM's final y update)
- Deep supervision allows progressive refinement over multiple steps
- ACT halting mechanism determines when to stop improving and move to next example
