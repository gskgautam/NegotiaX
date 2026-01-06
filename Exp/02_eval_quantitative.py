import sys
import os
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import config
from src.models.modeling_mac import HHHMistral
from src.evaluation.generate import Generator
from transformers import AutoTokenizer

def main():
    # Load Model
    print("Loading Model...")
    # Ideally load from checkpoint, here we init fresh for demo or load if exists
    # model = HHHMistral(config.BASE_MODEL_NAME, mac_layers=config.MAC_LAYERS)
    # model.load_state_dict(torch.load(os.path.join(config.OUTPUT_DIR, "checkpoint_epoch_3/hhh_mistral.pt")))
    
    # For the purpose of this script, we assume the model is loaded. 
    # If no checkpoint, we just warn.
    try:
        model = HHHMistral(config.BASE_MODEL_NAME, mac_layers=config.MAC_LAYERS, mac_head_dim=config.MAC_HEAD_DIM)
        checkpoint_path = os.path.join(config.OUTPUT_DIR, "checkpoint_epoch_3", "hhh_model.pt")
        
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}...")
            # Load to CPU first to avoid VRAM spike, let load_state_dict handle the move
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"Loaded checkpoint. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
            # We expect unexpected keys (base model QLoRA params) since we frozen them
            # We expect NO missing keys for 'mac_modules' 
            mac_missing = [k for k in missing if 'mac_modules' in k]
            if mac_missing:
                print(f"WARNING: Missing MAC keys: {mac_missing}")
            else:
                print("Confirmed: All MAC module weights loaded successfully.")
        else:
            print("No checkpoint found! Using initialized model (random MAC weights). This is likely not what you want.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error loading model: {e}")
        return

    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    generator = Generator(model, tokenizer)
    
    # Generate Outputs
    print("Generating Alpaca Responses...")
    generator.generate_alpaca(config.ALPACA_TEST_PATH, os.path.join(config.OUTPUT_DIR, "alpaca_results.json")) # Full dataset
    
    print("Generating BeaverTails Responses...")
    generator.generate_beaver(config.BEAVER_TEST_PATH, os.path.join(config.OUTPUT_DIR, "beaver_results.csv"))
    
    print("Generating TruthfulQA Responses...")
    generator.generate_truthful(config.TRUTHFUL_TEST_PATH, os.path.join(config.OUTPUT_DIR, "truthful_results.csv"))
    
    print("\nGeneration Complete.")
    print("="*50)
    print("INSTRUCTIONS FOR METRIC CALCULATION")
    print("="*50)
    print("1. Helpfulness (WR): Run GPT-Judge on 'outputs/alpaca_results.json'.")
    print("2. Harmlessness (SS): Run BeaverDam-7B on 'outputs/beaver_results.csv'.")
    print("3. Honesty (TI): Run GPT-Judge on 'outputs/truthful_results.csv'.")
    print("\nOnce metrics are obtained, fill in Table 1 below:")
    
    print("\nTable 1 – Alignment metrics for our model (MAC)")
    print("| Model        | WR ↑ | SS ↓ | TI ↑ | Avg ↑ |")
    print("|--------------|------|------|------|-------|")
    print("| MAC (Ours)   |      |      |      |       |")
    print("| RAHF (SOTA)* |      |      |      |       |")
    print("| Aligner*     |      |      |      |       |")
    print("| MARL-Focal*  |      |      |      |       |")
    print("| TrinityX*    |      |      |      |       |")
    print("| H3Fusion*    |      |      |      |       |")

if __name__ == "__main__":
    main()
