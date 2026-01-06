import sys
import os
import torch
import time
import psutil

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import config
from src.models.modeling_mac import HHHMistral
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def measure_throughput(model, tokenizer, input_text, num_runs=3):
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    start_time = time.time()
    total_tokens = 0
    
    with torch.no_grad():
        for _ in range(num_runs):
            outputs = model.generate(**inputs, max_new_tokens=50)
            total_tokens += 50 # Approx
            
    end_time = time.time()
    duration = end_time - start_time
    return total_tokens / duration

def main():
    print("Analyzing Base Model...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL_NAME, 
        quantization_config=bnb_config,
        device_map="auto"
    )
    # base_model.to(device)
    
    base_params = count_parameters(base_model)
    print(f"Base Model Params: {base_params / 1e9:.2f} B")
    
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME)
    throughput_base = measure_throughput(base_model, tokenizer, "Hello, how are you?")
    print(f"Base Inference Throughput: {throughput_base:.2f} tok/s")
    
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("\nAnalyzing MAC Model...")
    mac_model = HHHMistral(config.BASE_MODEL_NAME, mac_layers=config.MAC_LAYERS, mac_head_dim=config.MAC_HEAD_DIM)
  
    
    mac_params = count_parameters(mac_model)
    print(f"MAC Model Params: {mac_params / 1e9:.2f} B")
    print(f"Delta Params: {mac_params - base_params}")
    
    throughput_mac = measure_throughput(mac_model, tokenizer, "Hello, how are you?")
    print(f"MAC Inference Throughput: {throughput_mac:.2f} tok/s")
    
    
    print("\nTable 2 – Computational overhead")
    print("| Model        | Params (B) | ΔParams | Train Time/Epoch | Inference tok/s | VRAM (GB) |")
    print("|--------------|-----------:|--------:|------------------|-----------------|-----------|")
    print(f"| Base Model   | {base_params/1e9:.2f}       | 0       |                  | {throughput_base:.2f}            |           |")
    print(f"| MAC (Ours)   | {mac_params/1e9:.2f}       | {mac_params - base_params} |                  | {throughput_mac:.2f}            |           |")

if __name__ == "__main__":
    main()

