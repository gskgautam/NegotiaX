import sys
import os
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import config
from src.data.dataset import AlpacaDataset, BeaverTailsDataset, TruthfulQADataset
from src.models.modeling_mac import HHHMistral
from src.training.trainer import HHHTrainer
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, choices=["A1", "A2", "A3_HelpHarm", "A3_HelpHonest", "A3_HarmHonest"], required=True)
    args = parser.parse_args()
    
    print(f"Running Ablation: {args.variant}")
    
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load Datasets based on variant
    loaders = {}
    
    # Helpfulness
    if "A3_HarmHonest" not in args.variant:
        alpaca = AlpacaDataset(config.ALPACA_PATH, tokenizer, max_length=config.MAX_LENGTH)
        loaders['help'] = DataLoader(alpaca, batch_size=config.BATCH_SIZE, shuffle=True)
        
    # Harmlessness
    if "A3_HelpHonest" not in args.variant:
        beaver = BeaverTailsDataset(config.BEAVER_PATH, tokenizer, max_length=config.MAX_LENGTH, is_safe=True)
        loaders['harm'] = DataLoader(beaver, batch_size=config.BATCH_SIZE, shuffle=True)
        
    # Honesty
    if "A3_HelpHarm" not in args.variant:
        truthful = TruthfulQADataset(config.TRUTHFUL_PATH, tokenizer, max_length=config.MAX_LENGTH)
        loaders['honest'] = DataLoader(truthful, batch_size=config.BATCH_SIZE, shuffle=True)
        
    # Model Configuration
    mac_mode = "standard"
    if args.variant == "A1":
        mac_mode = "generic"
    elif args.variant == "A2":
        mac_mode = "frozen"
        
    print(f"Initializing Model with mode={mac_mode}...")
    model = HHHMistral(config.BASE_MODEL_NAME, mac_layers=config.MAC_LAYERS, mac_head_dim=config.MAC_HEAD_DIM, mac_mode=mac_mode)
    
    print("Starting Training...")
    trainer = HHHTrainer(model, loaders)
    trainer.train()
    
    print("Training Complete.")
    print("To evaluate, run experiments/02_eval_quantitative.py (modify it to load this checkpoint).")

if __name__ == "__main__":
    main()
