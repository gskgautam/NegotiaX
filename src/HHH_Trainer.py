import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import os
from ..config import config

class HHHTrainer:
    def __init__(self, model, train_loaders, test_loaders=None):
        self.model = model
        self.train_loaders = train_loaders # Dict: {'help': loader, 'harm': loader, 'honest': loader}
        self.test_loaders = test_loaders
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not (hasattr(config, 'USE_4BIT') and config.USE_4BIT):
             self.model.to(self.device)
        else:
             print("Skipping model.to(device) because QLoRA (4-bit) is enabled.")
        
    def train(self):
        optimizer = AdamW(self.model.parameters(), lr=config.LEARNING_RATE)
        
        
        min_len = min([len(l) for l in self.train_loaders.values()])
        total_steps = min_len * config.NUM_EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)
        
        self.model.train()
        
        for epoch in range(config.NUM_EPOCHS):
            print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
            
            # Create iterators
            iters = {k: iter(v) for k, v in self.train_loaders.items()}
            
            # progress_bar = tqdm(range(min_len))
            print(f"Starting epoch {epoch+1} loop...")
            
            for step in range(min_len):
                try:
                    total_loss = 0
                    optimizer.zero_grad()
                    
                    # Accumulate gradients from all 3 tasks
                    # We can do this sequentially to save memory (gradient accumulation)
                    
                    # 1. Helpfulness
                    batch_help = next(iters['help'])
                    loss_help = self._step(batch_help)
                    total_loss += loss_help.item()
                    loss_help.backward()
                    
                    # 2. Harmlessness
                    batch_harm = next(iters['harm'])
                    loss_harm = self._step(batch_harm)
                    total_loss += loss_harm.item()
                    loss_harm.backward()
                    
                    # 3. Honesty
                    batch_honest = next(iters['honest'])
                    loss_honest = self._step(batch_honest)
                    total_loss += loss_honest.item()
                    loss_honest.backward()
                    
                    optimizer.step()
                    scheduler.step()
                    
                    if step % 10 == 0:
                        print(f"Step {step}: Loss: {total_loss/3:.4f}", flush=True)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"Error at step {step}: {e}")
                    raise e
                
            # Save checkpoint
            save_path = os.path.join(config.OUTPUT_DIR, f"checkpoint_epoch_{epoch+1}")
            os.makedirs(save_path, exist_ok=True)
            self.model.save_pretrained(save_path)
            
    def _step(self, batch):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss


