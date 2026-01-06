import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    print("Loading Base Mistral Model...")
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        print("Model loaded. Running forward pass...")
        input_text = "Hello, world!"
        inputs = tokenizer(input_text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        print("Forward pass successful!")
        print(f"Logits shape: {outputs.logits.shape}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Base model failed: {e}")

if __name__ == "__main__":
    main()
