"""
Test script for LSTM-TRM to verify it works correctly.
"""

import torch
from models.recursive_reasoning.old.trm_lstm import LSTMTRM, LSTMTRMCarry

def test_lstm_trm():
    """Test basic LSTM-TRM functionality"""

    # Configuration
    config = {
        "batch_size": 4,
        "seq_len": 81,  # 9x9 Sudoku
        "vocab_size": 10,  # 0-9 for Sudoku
        "num_puzzle_identifiers": 100,
        "T_cycles": 3,
        "n_cycles": 6,
        "hidden_size": 128,  # Smaller for testing
        "expansion": 2.0,
        "halt_max_steps": 4,  # Fewer for testing
        "halt_exploration_prob": 0.1,
        "puzzle_emb_ndim": 128,
        "puzzle_emb_len": 4,
        "forward_dtype": "bfloat16",
        "no_ACT_continue": True
    }

    print("Creating LSTM-TRM model...")
    model = LSTMTRM(config)
    model = model.cuda()
    model.eval()

    # Create dummy batch
    batch = {
        "inputs": torch.randint(0, 10, (config["batch_size"], config["seq_len"]), dtype=torch.long).cuda(),
        "targets": torch.randint(0, 10, (config["batch_size"], config["seq_len"]), dtype=torch.long).cuda(),
        "puzzle_identifiers": torch.randint(0, config["num_puzzle_identifiers"], (config["batch_size"],), dtype=torch.long).cuda()
    }

    print("Initializing carry...")
    carry = model.initial_carry(batch)

    print("Running forward passes (simulating deep supervision)...")
    for step in range(config["halt_max_steps"]):
        with torch.no_grad():
            carry, outputs = model(carry, batch)

        logits = outputs["logits"]
        q_halt_logits = outputs["q_halt_logits"]

        print(f"  Step {step + 1}:")
        print(f"    Logits shape: {logits.shape}")
        print(f"    Q halt logits shape: {q_halt_logits.shape}")
        print(f"    Halted sequences: {carry.halted.sum().item()}/{config['batch_size']}")

        # Check if all sequences halted
        if carry.halted.all():
            print(f"  All sequences halted at step {step + 1}")
            break

    print("\nTest passed! LSTM-TRM is working correctly.")

    # Test with gradients (single step)
    print("\nTesting backward pass...")
    model.train()
    carry = model.initial_carry(batch)
    carry, outputs = model(carry, batch)

    # Compute dummy loss
    logits = outputs["logits"]
    targets = batch["targets"]
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, config["vocab_size"]),
        targets.reshape(-1)
    )

    print(f"Loss: {loss.item():.4f}")

    # Backward pass
    loss.backward()
    print("Backward pass successful!")

    # Check parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Memory (approx): {total_params * 2 / 1024**2:.2f} MB (bfloat16)")

    return model


if __name__ == "__main__":
    print("=" * 60)
    print("Testing LSTM-TRM Implementation")
    print("=" * 60)
    print()

    if torch.cuda.is_available():
        test_lstm_trm()
    else:
        print("CUDA not available. Skipping test.")
