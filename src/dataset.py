import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional
from transformers import PreTrainedTokenizer

class AlpacaDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(file_path, 'r') as f:
            self.data = json.load(f)
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        output_text = item.get('output', '')
        
        if input_text:
            prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
            
        full_text = prompt + output_text + self.tokenizer.eos_token
        
        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Create labels: mask out the prompt part for loss calculation if desired, 
        # but for simplicity in standard SFT, we often train on the whole sequence or just response.
        # Here we'll train on the whole sequence for simplicity, or we can implement masking.
        # Let's do standard CLM training for now.
        
        labels = encodings['input_ids'].clone()
        
        return {
            'input_ids': encodings['input_ids'].squeeze(0),
            'attention_mask': encodings['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0),
            'axis': 'helpfulness' # Tag for the collator/trainer if needed
        }

class BeaverTailsDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, max_length: int = 512, is_safe: bool = True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = pd.read_csv(file_path)
        
        # Filter for safe/unsafe if needed. The prompt says "27,186 safe pairs used for alignment training".
        # Assuming the CSV has a 'is_safe' column or similar. 
        # If not, we'll assume the provided train file is already filtered or we check columns.
        # Let's inspect the CSV columns first in a real scenario, but here I'll assume standard columns.
        # Common columns: prompt, response, is_safe
        
        if 'is_safe' in self.data.columns and is_safe:
             self.data = self.data[self.data['is_safe'] == True]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        prompt = row['prompt']
        response = row['response']
        
        # Simple QA format
        full_text = f"Question: {prompt}\nAnswer: {response}{self.tokenizer.eos_token}"
        
        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        labels = encodings['input_ids'].clone()
        
        return {
            'input_ids': encodings['input_ids'].squeeze(0),
            'attention_mask': encodings['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0),
            'axis': 'harmlessness'
        }

class TruthfulQADataset(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = pd.read_csv(file_path)
        # Columns usually: Question, Best Answer, Correct Answers, Incorrect Answers
        # We want to train on Question -> Best Answer (or Correct Answers)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        question = row['question']
        # Use 'answer' column directly from our preprocessed CSV
        answer = row['answer']
        
        full_text = f"Q: {question}\nA: {answer}{self.tokenizer.eos_token}"
        
        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        labels = encodings['input_ids'].clone()
        
        return {
            'input_ids': encodings['input_ids'].squeeze(0),
            'attention_mask': encodings['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0),
            'axis': 'honesty'
        }
