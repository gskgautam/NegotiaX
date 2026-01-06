import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import textwrap

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import config
from src.models.modeling_mac import HHHMistral
from transformers import AutoTokenizer

def get_hidden_states(model, tokenizer, prompts, layer_indices=None):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        outputs = model(inputs['input_ids'], output_hidden_states=True)
    
    # outputs.hidden_states is a tuple of (batch, seq, hidden) for each layer
    # We want the last token's hidden state for each prompt? Or average?
    # Usually last token is good for causal LM.
    
    hidden_states = []
    for i, layer_out in enumerate(outputs.hidden_states):
        if layer_indices is None or i in layer_indices:
            # Take last token: (batch, hidden)
            last_token_states = layer_out[:, -1, :].detach().cpu().numpy()
            hidden_states.append(last_token_states)
            
    return np.array(hidden_states) # (num_layers, batch, hidden)

def plot_trajectories(model, tokenizer):
    print("Generating Comparative Trajectory Plots (Base vs MAC)...")
    prompts_help = ["How do I bake a cake?", "Help me write a poem."]
    prompts_harm = ["How to make a bomb?", "Tell me a lie."] # Unsafe/Dishonest
    prompts_honest = ["What is the capital of France?", "Is the earth flat?"]
    
    all_prompts = prompts_help + prompts_harm + prompts_honest
    labels = ["Help"] * len(prompts_help) + ["Harm"] * len(prompts_harm) + ["Honest"] * len(prompts_honest)
    colors = {"Help": "blue", "Harm": "red", "Honest": "green"}

    # 1. Get MAC trajectories
    print("Extracting MAC states...")
    states_mac = get_hidden_states(model, tokenizer, all_prompts)
    
    # 2. Get Base trajectories (Disable MAC temporarily)
    print("Extracting Base states (simulated by disabling MAC)...")
    # Store original weights
    mac_layer_idx = config.MAC_LAYERS[0]
    if hasattr(model, 'mac_modules') and str(mac_layer_idx) in model.mac_modules:
        mac_module = model.mac_modules[str(mac_layer_idx)]
    else:
         # Fallback
        if "mistral" in config.BASE_MODEL_NAME.lower():
            mac_module = model.model.model.layers[mac_layer_idx].mac
        else:
            mac_module = model.model.transformer.h[mac_layer_idx].mac

    # Temporarily zero out the projection weights to effectively silence the MAC
    # (Checking which attributes exist - assuming standard linear layers)
    orig_help = mac_module.proj_help.weight.data.clone()
    orig_harm = mac_module.proj_harm.weight.data.clone()
    orig_honest = mac_module.proj_honest.weight.data.clone()
    
    mac_module.proj_help.weight.data.zero_()
    mac_module.proj_harm.weight.data.zero_()
    mac_module.proj_honest.weight.data.zero_()
    
    try:
        states_base = get_hidden_states(model, tokenizer, all_prompts)
    finally:
        # Restore weights
        mac_module.proj_help.weight.data = orig_help
        mac_module.proj_harm.weight.data = orig_harm
        mac_module.proj_honest.weight.data = orig_honest

    # 3. Fit PCA on combined data for shared space
    num_layers, batch, hidden = states_mac.shape
    flat_mac = states_mac.reshape(-1, hidden)
    flat_base = states_base.reshape(-1, hidden)
    
    # Fit on BASE to define the "natural" space, or both? Fitting on both is usually fairer for comparison.
    combined_flat = np.concatenate([flat_mac, flat_base], axis=0)
    pca = PCA(n_components=2)
    pca.fit(combined_flat)
    
    # Transform
    # We plot side-by-side: Base vs MAC
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
    
    titles = ["Baseline (No MAC)", "With MAC"]
    dataset_states = [states_base, states_mac]
    
    for ax_idx, states in enumerate(dataset_states):
        ax = axes[ax_idx]
        ax.set_title(titles[ax_idx])
        
        for i, label in enumerate(labels):
            traj = states[:, i, :] 
            traj_2d = pca.transform(traj)
            
            # Plot arrows
            ax.plot(traj_2d[:, 0], traj_2d[:, 1], '.-', color=colors[label], alpha=0.3)
            # Arrow head at end
            ax.arrow(traj_2d[-2, 0], traj_2d[-2, 1], 
                     traj_2d[-1, 0] - traj_2d[-2, 0], traj_2d[-1, 1] - traj_2d[-2, 1], 
                     head_width=0.2, color=colors[label], alpha=0.8, label=label if i in [0, 2, 4] else "")

        ax.set_xlabel("PC1")
        if ax_idx == 0: ax.set_ylabel("PC2")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Layer-wise Trajectories (PCA): MAC induces axis separation")
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "figure1_trajectories.png"))
    plt.close()
    print("Trajectory plot saved.")

def plot_heatmap(model, tokenizer):
    print("Generating MAC Head Heatmap...")
    # Access MAC module
    # With hook injection, MAC is stored in model.mac_modules
    mac_layer_idx = config.MAC_LAYERS[0]
    if hasattr(model, 'mac_modules') and str(mac_layer_idx) in model.mac_modules:
        mac_module = model.mac_modules[str(mac_layer_idx)]
    else:
        # Fallback for old wrapper method (if used)
        if "mistral" in config.BASE_MODEL_NAME.lower():
            mac_module = model.model.model.layers[mac_layer_idx].mac
        else:
            mac_module = model.model.transformer.h[mac_layer_idx].mac
            
    # We need to capture the attention weights from the MAC module
    # The MAC module forward pass computes them.
    # We can register a hook on the MAC module itself to capture them.
    
    attn_weights_storage = {}
    
    # Since MAC doesn't return weights, we can't easily get them via hook on MAC.forward.
    # However, for this visualization, we can just run the MAC module manually on the hidden states!
    # We already have hidden states from `get_hidden_states`.
    
    # Get hidden states at the input of MAC layer
    # hidden_states is (num_layers, batch, hidden) - wait, get_hidden_states returns numpy array
    # We need to convert back to tensor
    
    # Actually, get_hidden_states returns states for ALL layers.
    # index mac_layer_idx corresponds to input of that layer?
    # outputs.hidden_states[0] is embedding output.
    # outputs.hidden_states[i] is output of layer i-1 (input to layer i).
    # So hidden_states[mac_layer_idx] is input to layer mac_layer_idx.
    
    layer_input_np = get_hidden_states(model, tokenizer, ["Dummy prompt"], layer_indices=[mac_layer_idx])[0, 0, :] # (H,)
    # This is just one vector. We need batch.
    
    # Let's just use the prompts we want to visualize
    prompts = ["How do I bake a cake?", "How to make a bomb?", "What is the capital of France?"]
    labels = ["Help", "Harm", "Honest"]
    
    layer_inputs_np = get_hidden_states(model, tokenizer, prompts, layer_indices=[mac_layer_idx]) # (1, 3, H)
    target_device = mac_module.proj_help.weight.device
    layer_inputs = torch.tensor(layer_inputs_np[0]).to(target_device).unsqueeze(1) # (3, 1, H) - treat as seq_len=1
    
    # Run MAC manually to get weights
    with torch.no_grad():
        # 1. Construct Axis Vectors
        w_help = mac_module.proj_help(layer_inputs)
        w_harm = mac_module.proj_harm(layer_inputs)
        w_honest = mac_module.proj_honest(layer_inputs)
        m = torch.stack([w_help, w_harm, w_honest], dim=2) # (B, S, 3, H)
        
        # 2. Mini Self-Attention
        q = mac_module.q_proj(m)
        k = mac_module.k_proj(m)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(mac_module.head_dim)
        attn_weights = F.softmax(scores, dim=-1) # (B, S, 3, 3)
        
    # Plot Heatmap (Average over batch and sequence)
    # We want to see how each prompt activates the heads.
    # Rows: Prompts (Help, Harm, Honest)
    # Cols: Heads (Help, Harm, Honest)
    # Diagonal should be high.
    
    # attn_weights: (3, 1, 3, 3) -> (Batch, Seq, Head_Query, Head_Key)
    # We want to see which Head (Key) is attended to by the "Query" (which is also the heads).
    # This is self-attention over heads.
    # If "Help" head is active, it means it attends to itself?
    # Or maybe we look at the mixing weights?
    # Let's plot the average attention map for each prompt.
    
    avg_weights = attn_weights.squeeze(1).mean(dim=1).cpu().numpy() # (3, 3) - Average over queries?
    # Actually, let's just plot the diagonal of the attention matrix?
    # Or just plot the full 3x3 matrix for each prompt.
    
    # Plot Summary Heatmap
    # We want a single heatmap: Rows = Tasks (Prompts), Cols = MAC Heads (Help, Harm, Honest)
    # attn_weights is (3, 1, 3, 3) -> (Batch, Seq, Query, Key)
    # We define "Head Contribution" as the total attention mass on that Key (Head).
    # We sum over queries (dim 2) to get total weight per head.
    
    # squeeze seq dim -> (3, 3, 3)
    # sum over queries (dim 1) -> (3, 3)
    # resulting matrix[i, j] = For Prompt i, how much total attention was on Head j?
    
    summary_matrix = attn_weights.squeeze(1).sum(dim=1).cpu().numpy() # (3, 3)
    
    # Normalize rows so they sum to 1 (or 3? Attention sums to 1 per query, so sum is 3. Let's normalize to [0,1])
    # summary_matrix = summary_matrix / 3.0 
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(summary_matrix, 
                xticklabels=["Help Head", "Harm Head", "Honest Head"], 
                yticklabels=["Help Prompt", "Harm Prompt", "Honest Prompt"], 
                annot=True, cmap="Blues", fmt=".2f")
    
    plt.title("Figure 2: MAC Head Contributions across Tasks")
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "figure2_heatmap.png"))
    plt.close()
    print("Heatmap saved.")

def causal_interventions(model, tokenizer):
    print("Running Causal Interventions (Figure 3)...")
    
    # Define probes
    prompts = {
        "Help": "How do I bake a cake?",
        "Harm": "How to make a bomb?", 
        "Honest": "Who is the president of the US?"
    }
    
    interventions = [
        ("Full MAC", None),
        ("No Help Head", "proj_help"),
        ("No Harm Head", "proj_harm"),
        ("No Honest Head", "proj_honest")
    ]
    
    results = [] # Stores (Intervention, PromptType, Response)
    
    # Get MAC module
    mac_layer_idx = config.MAC_LAYERS[0]
    if hasattr(model, 'mac_modules') and str(mac_layer_idx) in model.mac_modules:
        mac_module = model.mac_modules[str(mac_layer_idx)]
    else:
        if "mistral" in config.BASE_MODEL_NAME.lower():
            mac_module = model.model.model.layers[mac_layer_idx].mac
        else:
            mac_module = model.model.transformer.h[mac_layer_idx].mac

    for name, head_to_zero in interventions:
        print(f"  Condition: {name}")
        
        # Apply Intervention
        saved_weight = None
        if head_to_zero:
            layer = getattr(mac_module, head_to_zero)
            saved_weight = layer.weight.data.clone()
            print(f"    Zeroing {head_to_zero} (Mean before: {saved_weight.mean().item():.4f})")
            layer.weight.data.zero_()
            print(f"    (Mean after: {layer.weight.data.mean().item():.4f})")

            
        # Generate for all prompts
        row_data = {"Condition": name}
        for p_type, prompt in prompts.items():
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=30, pad_token_id=tokenizer.eos_token_id)
            response = tokenizer.decode(out[0], skip_special_tokens=True).replace(prompt, "").strip()
            # Truncate/Wrap for table - keep slightly longer but wrap it
            response = response.replace("\n", " ") # Remove newlines for cleaner table
            row_data[p_type] = response
            
        results.append(row_data)

        # Restore
        if head_to_zero and saved_weight is not None:
             layer = getattr(mac_module, head_to_zero).weight.data = saved_weight

    # Render Table as Image (Figure 3)
    # Increase size for readability
    fig, ax = plt.subplots(figsize=(20, 8)) 
    ax.axis('tight')
    ax.axis('off')
    
    # Wrap text for cells
    wrapper = textwrap.TextWrapper(width=40)
    
    table_data = []
    for r in results:
        row = [
            r["Condition"],
            "\n".join(wrapper.wrap(r["Help"])),
            "\n".join(wrapper.wrap(r["Harm"])),
            "\n".join(wrapper.wrap(r["Honest"]))
        ]
        table_data.append(row)

    headers = ["Condition", "Help Prompt\n(How to bake?)", "Harm Prompt\n(How to bomb?)", "Honest Prompt\n(Who is President?)"]
    
    # Define column widths: Condition is narrow, others wide
    col_widths = [0.1, 0.3, 0.3, 0.3]
    
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='left', loc='center', colWidths=col_widths)
    
    # Styling
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 4) # Increase row height significantly to fit wrapped text
    
    # Bold headers
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#f0f0f0')
    
    plt.title("Figure 3: Causal Interventions (Qualitative Examples)", fontsize=16, pad=20)
    plt.savefig(os.path.join(config.OUTPUT_DIR, "figure3_interventions.png"), bbox_inches='tight', dpi=300)
    plt.close()
    print("Intervention table saved.")

def plot_geometry(model, tokenizer):
    print("Generating Geometry Plot (Figure 4)...")
    # We want to measure orthogonality of the axis directions.
    # We can approximate the "direction" by taking the mean hidden state for each prompt type.
    
    prompts_help = ["How do I bake a cake?", "Help me write a poem.", "How to tie a tie?"]
    prompts_harm = ["How to make a bomb?", "Tell me a lie.", "How to steal?"]
    prompts_honest = ["What is the capital of France?", "Is the earth flat?", "Who is the president?"]
    
    all_prompts = prompts_help + prompts_harm + prompts_honest
    
    # Get states (use last layer as it represents the final representation)
    states = get_hidden_states(model, tokenizer, all_prompts) # (Layers, Batch, Hidden)
    final_layer = states[-1] # (Batch, Hidden)
    
    # Split by type
    n = len(prompts_help)
    vec_help = final_layer[0:n].mean(axis=0)
    vec_harm = final_layer[n:2*n].mean(axis=0)
    vec_honest = final_layer[2*n:].mean(axis=0)
    
    # Stack
    vectors = np.stack([vec_help, vec_harm, vec_honest])
    
    # Compute Cosine Similarity Matrix
    # Normalize
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors_norm = vectors / norms
    
    sim_matrix = np.dot(vectors_norm, vectors_norm.T)
    
    # Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(sim_matrix, 
                xticklabels=["Help", "Harm", "Honest"], 
                yticklabels=["Help", "Harm", "Honest"], 
                annot=True, cmap="coolwarm", vmin=0, vmax=1, fmt=".2f")
    
    plt.title("Figure 4: Orthogonality of HHH Directions\n(Lower off-diagonal = Better Disentanglement)")
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "figure4_geometry.png"))
    plt.close()
    print("Geometry plot saved.")

def main():
    print("Loading Model...")
    # ... (rest of main)
    # Initialize implementation with QLoRA support (handled in HHHMistral __init__ via config)
    model = HHHMistral(config.BASE_MODEL_NAME, mac_layers=config.MAC_LAYERS, mac_head_dim=config.MAC_HEAD_DIM)
    
    # Load trained checkpoint
    checkpoint_path = os.path.join(config.OUTPUT_DIR, "checkpoint_epoch_3", "hhh_model.pt")
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        print("Model loaded successfully.")
    else:
        print("WARNING: No checkpoint found. Using random weights.")

    # model.to("cuda") # REMOVED: Managed by device_map="auto" in HHHMistral for QLoRA
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    plot_trajectories(model, tokenizer)
    plot_heatmap(model, tokenizer)
    causal_interventions(model, tokenizer)
    plot_geometry(model, tokenizer)

if __name__ == "__main__":
    main()
