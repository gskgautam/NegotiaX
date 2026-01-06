import torch
from tqdm import tqdm
import json
import pandas as pd

class Generator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model.to(self.device) # Don't move model if QLoRA/auto map is used
        self.model.eval()
        
    @property
    def device(self):
        return self.model.device
        
    def generate_alpaca(self, test_file, output_file, limit=None):
        with open(test_file, 'r') as f:
            data = json.load(f)
        
        if limit:
            data = data[:limit]
            
        results = []
        for item in tqdm(data, desc="Generating Alpaca"):
            instruction = item.get('instruction', '')
            input_text = item.get('input', '')
            
            if input_text:
                prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
            else:
                prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
                
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=256)
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract response part
            if "### Response:\n" in response:
                response = response.split("### Response:\n")[1]
                
            results.append({
                "instruction": instruction,
                "input": input_text,
                "generated_response": response
            })
            
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
    def generate_beaver(self, test_file, output_file, limit=None):
        df = pd.read_csv(test_file)
        if limit:
            df = df.iloc[:limit]
            
        results = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating BeaverTails"):
            prompt = row['prompt']
            # Simple QA format
            full_prompt = f"Question: {prompt}\nAnswer:"
            
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=256)
                
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "Answer:" in response:
                response = response.split("Answer:")[1]
                
            results.append({
                "prompt": prompt,
                "generated_response": response,
                "is_safe_label": row.get('is_safe', None)
            })
            
        pd.DataFrame(results).to_csv(output_file, index=False)

    def generate_truthful(self, test_file, output_file, limit=None):
        df = pd.read_csv(test_file)
        if limit:
            df = df.iloc[:limit]
            
        results = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating TruthfulQA"):
            question = row['question']
            full_prompt = f"Q: {question}\nA:"
            
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=256)
                
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "A:" in response:
                response = response.split("A:")[1]
                
            results.append({
                "Question": question,
                "generated_response": response
            })
            
        pd.DataFrame(results).to_csv(output_file, index=False)
