# Mechanistic Alignment Circuit (MAC) for Mistral-7B

## 1. Project Overview: What Are We Doing and Why?
**The Goal:**
Large Language Models (LLMs) like Mistral-7B are powerful but can sometimes generate harmful, untruthful, or unhelpful content. Traditional "alignment" (making models safe and helpful) usually involves **Fine-Tuning (RLHF)**, which updates *all* or *most* of the model's billions of parameters. This is expensive, slow, and computationally heavy.

**Our Approach:**
Instead of retraining the whole brain, we inject a small **Mechanistic Alignment Circuit (MAC)** into the model's "thought process" (specifically at Layer 15). Think of this as adding a "conscience module" that monitors the model's internal thoughts and nudges them towards:
1.  **Helpfulness**: Giving correct, useful instructions.
2.  **Harmlessness**: Refusing to generate dangerous content (bombs, poisons, etc.).
3.  **Honesty**: Avoiding hallucinations and lies.

**The "How":**
We freeze the giant base model (Mistral-7B) so it doesn't change. We only train the tiny MAC module (plus some lightweight LoRA adapters). This allows us to "align" a massive 7B model on a single consumer GPU in just a few hours.

## 2. Technical Methodology
### A. The MAC Mechanism (The "Conscience")
The MAC module is a custom neural network block inserted into the transformer. It works in three steps:
1.  **Projections (The "Sensors")**: It looks at the model's current hidden state and projects it onto three semantic axes: Helpfulness, Harmlessness, and Honesty. It asks: *"Is the current thought helpful? Is it safe? Is it true?"*
2.  **Mini-Attention (The "Decision")**: It uses a small attention mechanism to weigh these factors. If the model is drifting towards "Harmful," the Harmlessness head activates strongly.
3.  **Steering (The "Nudge")**: It adds a correction vector back into the residual stream. If the model was about to say something bad, the MAC pushes the hidden state vector towards "safety," changing the output token.

### B. QLoRA Optimization (How We Fit It in Memory)
Training a 7B model normally requires ~100GB+ of VRAM. We fit it into ~24GB using **QLoRA**:
*   **4-Bit Quantization**: We load the base Mistral model in 4-bit mode (compressing weights by 8x).
*   **LoRA**: We only learn low-rank matrices and the MAC weights.
*   **Result**: We train <2% of the total parameters, achieving similar results to full training with a fraction of the cost.

## 3. Implementation Details
*   **Base Model**: `mistralai/Mistral-7B-v0.1`
*   **Datasets Used**:
    *   **Helpfulness**: Stanford Alpaca (Instruction following).
    *   **Harmlessness**: BeaverTails (Safety alignment).
    *   **Honesty**: TruthfulQA (Fact-checking).
*   **Training Configuration**:
    *   **Epochs**: 3
    *   **Batch Size**: 2 (with gradient accumulation).
    *   **Optimizer**: AdamW.
    *   **Loss Function**: Cross-Entropy (standard next-token prediction).

## 4. Final Results (Research Metrics)
We evaluated the model on the full test sets (~5000+ samples) and graded the responses using `gpt-4o-mini` as an independent judge.

### A. Alignment Scores
| Metric | Dataset | Score | Interpretation |
```bash
pip install torch transformers peft bitsandbytes accelerate pandas scikit-learn seaborn matplotlib openai
```

### Step 2: Training (The Learning Phase)
Run the training script to teach the MAC module:
```bash
python experiments/01_train_mac.py
```
*   *Output*: Saves model weights to `outputs/checkpoint_epoch_3`.

### Step 3: Evaluation (The Testing Phase)
Generate responses for the 5000+ test questions:
```bash
python experiments/02_eval_quantitative.py
```
*   *Output*: Creates response files in `outputs/`.

### Step 4: Grading (The Scoring Phase)
Use GPT-4 as a teacher to grade the student model:
```bash
export OPENAI_API_KEY="your-key-here"
python experiments/04_grade_results_gpt4.py
```
*   *Output*: Prints the final research metrics table (as seen in Section 4).

### Step 5: Visualization (Optional)
See the "thoughts" of the model:
```bash
python experiments/05_mech_interp.py
```
*   *Output*: Generates heatmaps and trajectory plots in `outputs/`.

## 6. Directory Structure
*   `src/models/modeling_mac.py`: The brain of the MAC module.
*   `src/training/trainer.py`: The training loop logic.
*   `experiments/`: Scripts for running the pipeline step-by-step.
*   `outputs/`: All your results (models, logs, charts) live here.
