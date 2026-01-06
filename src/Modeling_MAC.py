import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from .mac import MechanisticAlignmentCircuit
from typing import List, Optional, Tuple, Union

# Import specific layer classes for type hinting and wrapping
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from ..config import config

class HHHMistral(nn.Module):
    def __init__(self, model_name: str, mac_layers: List[int], mac_head_dim: int = 64, mac_mode: str = "standard"):
        super().__init__()
        
        # QLoRA / 4-bit loading
        bnb_config = None
        if hasattr(config, 'USE_4BIT') and config.USE_4BIT:
            print("Loading model in 4-bit mode (QLoRA)...")
            compute_dtype = getattr(torch, config.BNB_4BIT_COMPUTE_DTYPE)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=config.BNB_4BIT_QUANT_TYPE,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=config.BNB_4BIT_USE_DOUBLE_QUANT,
            )
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto" if bnb_config else None
        )
        
        if bnb_config:
            self.model = prepare_model_for_kbit_training(self.model)
    
            
        self.config = self.model.config
        self.mac_layers = mac_layers
        self.mac_modules = nn.ModuleDict() # Store MAC modules
        
        # Detect model type
        if "mistral" in model_name.lower():
            self._inject_mistral(mac_layers, mac_head_dim, mac_mode)
        elif "gpt2" in model_name.lower():
            self._inject_gpt2(mac_layers, mac_head_dim, mac_mode)
        else:
            print(f"Warning: Model {model_name} not explicitly supported for MAC injection. Running as base model.")

    def _inject_mistral(self, mac_layers, mac_head_dim, mac_mode):
        for layer_idx in mac_layers:
            if 0 <= layer_idx < len(self.model.model.layers):
                # Create MAC module
                mac = MechanisticAlignmentCircuit(self.config.hidden_size, head_dim=mac_head_dim, mode=mac_mode)
                self.mac_modules[str(layer_idx)] = mac
                
                
                layer = self.model.model.layers[layer_idx]
                layer.self_attn.register_forward_hook(self._get_mac_hook(mac))
                print(f"Injected MAC at layer {layer_idx} via hook.")

    def _inject_gpt2(self, mac_layers, mac_head_dim, mac_mode):
        for layer_idx in mac_layers:
            if 0 <= layer_idx < len(self.model.transformer.h):
                mac = MechanisticAlignmentCircuit(self.config.hidden_size, head_dim=mac_head_dim, mode=mac_mode)
                self.mac_modules[str(layer_idx)] = mac
                
                layer = self.model.transformer.h[layer_idx]
                # GPT2Block: attn -> residual -> ln_2 -> mlp
                # Hook attn
                layer.attn.register_forward_hook(self._get_mac_hook(mac))
                print(f"Injected MAC at layer {layer_idx} via hook.")

    def _get_mac_hook(self, mac_module):
        def hook(module, input, output):
            # output is typically (hidden_states, present_key_value, ...)
            # We want to modify hidden_states
            
            if isinstance(output, tuple):
                hidden_states = output[0]
                rest = output[1:]
            else:
                hidden_states = output
                rest = ()
            
            
            if mac_module.axis_embeddings.device != hidden_states.device:
                mac_module.to(hidden_states.device)
            
            mac_out = mac_module(hidden_states)
            new_hidden_states = hidden_states + mac_out
            
            if isinstance(output, tuple):
                return (new_hidden_states,) + rest
            else:
                return new_hidden_states
        return hook

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
    
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)
    
    def save_pretrained(self, path):
        torch.save(self.state_dict(), f"{path}/hhh_model.pt")
        self.config.save_pretrained(path)

    @property
    def device(self):
        return self.model.device

