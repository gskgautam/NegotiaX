# NegotiaX

## 📚 Datasets

The following datasets can be accessed from their respective official sources:

- **Alpaca**  
  A strong instruction-following dataset built on top of the Stanford Self-Instruct method.  
  [Access Alpaca](https://github.com/tatsu-lab/stanford_alpaca)

- **BeaverTails**  
  A challenging benchmark designed for evaluating long-context instruction tuning.  
  [Access BeaverTails](https://sites.google.com/view/pku-beavertails)

- **TruthfulQA**  
  A benchmark to measure whether language models produce truthful answers.  
  [Access TruthfulQA](https://github.com/sylinrl/TruthfulQA)


## 🧠 Instruction-Tuned Models

The following instruction-tuned large language models can be downloaded from Hugging Face:

- **LLaMA-2 7B**  
  [Download from Hugging Face](https://huggingface.co/meta-llama/Llama-2-7b-hf)

- **Mistral-7B**  
  [Download from Hugging Face](https://huggingface.co/mistralai/Mistral-7B-v0.1)

 - **Gemma-7B**  
  [Download from Hugging Face](https://huggingface.co/google/gemma-7b)

- **DeepSeek-7B**  
  [Download from Hugging Face](https://huggingface.co/deepseek-ai/deepseek-llm-7b-base)

## Module Usage

- **`Exp/`**  
  Contains experimental scripts used to reproduce the main results, ablations, and analyses reported in the paper.

  - `Ablation.py`  
    Runs ablation studies for NegotiaX, isolating the effects of Module I (1-to-N objective separation) and Module II (Mechanistic Alignment Circuit).

Some experimental scripts (e.g., parts of `Ablation.py`) are **not shared in full** in this repository.

  - `Computational_Analysis.py`  
    Computes inference-time latency and peak GPU memory usage across base models, 1-to-N variants, and full NegotiaX.

  - `Mech_Interp.py`  
    Performs mechanistic interpretability analysis to study attention heads, objective-specific pathways, catastrophic forgetting, and mechanistic blind-spots.

  - `Train_MAC.py`  
    Training script for the Mechanistic Alignment Circuit (MAC), enabling learned negotiation across alignment objectives.

  - `test_base_model.py`  
    Evaluates standard 1-to-1 transformer baselines without NegotiaX for comparison.

---

- **`src/`**  
  Core implementation of the NegotiaX architecture and training pipeline.

  - `Dataset_Response_Generate.py`  
    Generates model responses for evaluation datasets (Alpaca, BeaverTails, TruthfulQA).

  - `HHH_Trainer.py`  
    Implements the training loop for objective-specific supervision and joint Helpful–Harmless–Honest (HHH) alignment.

  - `Load_Preprocess_Dataset.py`  
    Loads and preprocesses datasets corresponding to Helpfulness, Harmlessness, and Honesty objectives.

  - `MAC.py`  
    Implements the **Mechanistic Alignment Circuit (Module II)**, including objective-specific subspace projection and negotiative fusion.

  - `Modeling_MAC.py`  
    Defines the **NegotiaX 1-to-N transformer architecture**, integrating Module I (objective-specific representational pathways) and Module II (MAC).

### Logs

- **Logs.**  
  We only release log files that do **not contain any author-identifying information**. 


### 📈 Evaluation

After applying any of the MoCaE methods, use `Evaluate.py` to assess the performance of the calibrated models.

> ⚠️ Note: Make sure you have the appropriate access to the moderation models used for evaluation. These include:

- GPT-4.0 (via OpenAI API)
- beaver-dam-7b — available here: [PKU-Alignment/beaver-dam-7b](https://huggingface.co/PKU-Alignment/beaver-dam-7b)
- GPT-Judge (via OpenAI API) 

These evaluators are used to provide automated and/or human-aligned judgment of the calibrated outputs in terms of helpfulness, harmlessness, and honesty.

**Evaluation Protocol.**  
  Our evaluation strictly follows the standardized implementation used in prior multi-objective alignment work. In particular, we adopt the evaluation pipeline from the H³Fusion repository:
  https://github.com/sftekin/h3fusion/tree/main/evaluator

 

