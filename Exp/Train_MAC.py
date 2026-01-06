import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import config
from src.data.dataset import AlpacaDataset, BeaverTailsDataset, TruthfulQADataset
from src.models.modeling_mac import HHHMistral
from src.training.trainer import HHHTrainer
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

def main():
    print("Initializing Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading Datasets...")
    # Helpfulness
    alpaca = AlpacaDataset(config.ALPACA_PATH, tokenizer, max_length=config.MAX_LENGTH)
    loader_help = DataLoader(alpaca, batch_size=config.BATCH_SIZE, shuffle=True)
    
    # Harmlessness
    beaver = BeaverTailsDataset(config.BEAVER_PATH, tokenizer, max_length=config.MAX_LENGTH, is_safe=True)
    loader_harm = DataLoader(beaver, batch_size=config.BATCH_SIZE, shuffle=True)
    
    # Honesty
    truthful = TruthfulQADataset(config.TRUTHFUL_PATH, tokenizer, max_length=config.MAX_LENGTH)
    loader_honest = DataLoader(truthful, batch_size=config.BATCH_SIZE, shuffle=True)
    
    train_loaders = {
        'help': loader_help,
        'harm': loader_harm,
        'honest': loader_honest
    }
    
    print("Initializing Model...")
    model = HHHMistral(config.BASE_MODEL_NAME, mac_layers=config.MAC_LAYERS, mac_head_dim=config.MAC_HEAD_DIM)
    
    
    print("Starting Training...")
    trainer = HHHTrainer(model, train_loaders)
    trainer.train()
    
    print("Training Complete. Model saved.")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()

