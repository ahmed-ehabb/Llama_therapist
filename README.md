# 🧠 LLaMA Therapist

<div align="center">

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow.svg)

**A fine-tuned conversational AI chatbot powered by Meta's LLaMA 3.2, designed to provide empathetic and supportive therapeutic responses.**

[Features](#-features) •
[Demo](#-demo) •
[Installation](#-installation) •
[Usage](#-usage) •
[Model Details](#-model-details) •
[Training](#-training) •
[License](#-license)

</div>

---

## 📖 Overview

**LLaMA Therapist** is a specialized language model fine-tuned for therapeutic conversations. It leverages the power of **Meta's LLaMA 3.2 (3B Instruct)** with efficient **LoRA (Low-Rank Adaptation)** fine-tuning and **4-bit quantization** to deliver empathetic, context-aware responses to mental health and wellness queries.

This project demonstrates how modern LLMs can be adapted for domain-specific applications using parameter-efficient fine-tuning techniques, achieving professional-grade results on consumer hardware.

## ✨ Features

- 🎯 **Empathetic Responses**: Trained on 20k+ synthetic therapeutic conversations
- ⚡ **Efficient Fine-tuning**: Uses LoRA for parameter-efficient adaptation (~30% VRAM savings)
- 🚀 **Fast Inference**: Native 2x faster inference through Unsloth optimization
- 💬 **Multi-topic Support**: Handles conversations about:
  - Anxiety and stress management
  - Motivation and productivity
  - Relationship challenges
  - Personal growth and self-improvement
  - Loneliness and social connection
  - Time management strategies
- 🔧 **Quantized Model**: 4-bit quantization for reduced memory footprint
- 📊 **Comprehensive Evaluation**: Validated using ROUGE, BERTScore, and METEOR metrics

## 🎬 Demo

### Example Conversations

<details>
<summary><b>💭 Dealing with Anxiety</b></summary>

**User**: "I've been feeling really anxious lately, especially about work. What can I do?"

**LLaMA Therapist**: *Provides empathetic and actionable advice on managing workplace anxiety, including breathing techniques, breaking tasks into smaller steps, and the importance of self-care.*

</details>

<details>
<summary><b>💪 Finding Motivation</b></summary>

**User**: "I'm struggling to stay motivated with my projects. Any tips?"

**LLaMA Therapist**: *Offers strategies for maintaining motivation, setting realistic goals, celebrating small wins, and finding intrinsic motivation.*

</details>

<details>
<summary><b>❤️ Relationship Support</b></summary>

**User**: "How can I reconnect with an old friend I haven't talked to in years?"

**LLaMA Therapist**: *Provides thoughtful guidance on reaching out, overcoming hesitation, and rebuilding meaningful connections.*

</details>

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training/inference)
- 8GB+ VRAM (for inference with quantization)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/LLama_therapist.git
cd LLama_therapist
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download the model weights**

Due to GitHub's file size limitations, the fine-tuned model weights are hosted on Google Drive:

📦 **[Download Model Weights](https://drive.google.com/drive/folders/1AJD53yBGqSYZLwXeVFN4uUw65Mjw3h6f?usp=sharing)**

After downloading, extract the contents into the `lora_model1/` directory.

## 💻 Usage

### Quick Start

Open the `llama_therapist.ipynb` notebook and run the inference cells:

```python
from unsloth import FastLanguageModel
import torch

# Load the model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="lora_model1",
    max_seq_length=150,
    dtype=None,
    load_in_4bit=True,
)

# Enable inference mode for 2x faster generation
FastLanguageModel.for_inference(model)

# Create a conversation
messages = [
    {"role": "user", "content": "I've been feeling stressed about my workload. Any advice?"}
]

# Generate response
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=150,
    temperature=1.5,
    min_p=0.1,
    use_cache=True,
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Batch Inference

```python
# Multiple queries at once
queries = [
    "How can I manage my time better?",
    "I feel lonely. What should I do?",
    "Tips for staying focused?"
]

# Process in batches for efficiency
for query in queries:
    messages = [{"role": "user", "content": query}]
    # ... (same inference code as above)
```

## 🧬 Model Details

### Architecture

- **Base Model**: `unsloth/llama-3.2-3b-instruct-bnb-4bit`
- **Model Type**: Causal Language Model (Decoder-only Transformer)
- **Parameters**: 3 Billion (base) + ~2.4M trainable LoRA parameters
- **Quantization**: 4-bit (bitsandbytes)
- **Context Length**: 150 tokens (fine-tuned), 128k tokens (base capability)

### LoRA Configuration

| Parameter | Value |
|-----------|-------|
| Rank (r) | 16 |
| Alpha | 16 |
| Dropout | 0 |
| Target Modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Trainable Params | ~2.4M (~0.08% of base model) |

### Performance

| Metric | Score |
|--------|-------|
| ROUGE-1 | Evaluated on validation set |
| ROUGE-L | Evaluated on validation set |
| BERTScore F1 | Semantic similarity metric |
| METEOR | Translation quality metric |

*Detailed metrics available in the notebook.*

## 🎓 Training

### Dataset

- **Source**: [NART 100k Synthetic Dataset](https://huggingface.co/datasets/jerryjalapeno/nart-100k-synthetic)
- **Size**: ~20,000 conversation pairs (20% sample)
- **Format**: Vicuna-style multi-turn conversations
- **Split**: 99.5% train / 0.5% validation

### Training Configuration

```python
# Key hyperparameters
max_seq_length = 150
per_device_train_batch_size = 2
gradient_accumulation_steps = 4
learning_rate = 2e-4
num_train_epochs = 1
max_steps = 48
optimizer = "adamw_8bit"
lr_scheduler_type = "linear"
```

### Training Infrastructure

- **Framework**: Unsloth + HuggingFace Transformers
- **Trainer**: SFTTrainer (Supervised Fine-Tuning)
- **Optimization**: 4-bit quantization, LoRA adapters
- **Hardware**: GPU with CUDA support (training completed in ~48 steps)

### Reproduce Training

To reproduce the training process:

1. Open `llama_therapist.ipynb`
2. Run all cells sequentially
3. The fine-tuned model will be saved to `lora_model1/`

Training takes approximately 30-60 minutes on a modern GPU (e.g., RTX 3090, A100).

## 📚 Project Structure

```
LLama_therapist/
├── llama_therapist.ipynb      # Main training & inference notebook
├── lora_model1/                # Fine-tuned LoRA model weights
│   ├── adapter_config.json     # LoRA configuration
│   ├── adapter_model.safetensors  # Model weights (download separately)
│   ├── tokenizer.json          # Tokenizer (download separately)
│   ├── tokenizer_config.json
│   └── special_tokens_map.json
├── requirements.txt            # Python dependencies
├── .gitignore
└── README.md
```

## 🔧 Dependencies

### Core Libraries

- `torch` - PyTorch deep learning framework
- `transformers` - HuggingFace Transformers
- `unsloth` - Efficient LLM fine-tuning (~30% faster, less VRAM)
- `peft` - Parameter-Efficient Fine-Tuning (LoRA)
- `trl` - Transformers Reinforcement Learning
- `bitsandbytes` - 4-bit quantization
- `accelerate` - Distributed training support

### Data & Evaluation

- `datasets` - HuggingFace Datasets
- `pandas` - Data manipulation
- `scikit-learn` - Train/test split
- `rouge-score` - ROUGE metrics
- `bert-score` - BERTScore evaluation
- `evaluate` - Model evaluation metrics (METEOR)

See `requirements.txt` for the complete list with versions.

## 🛡️ Ethical Considerations

**Important Notice**: This model is a **research project** and should **NOT** be used as a replacement for professional mental health services.

- ⚠️ Not a substitute for licensed therapists or counselors
- ⚠️ May generate incorrect or inappropriate responses
- ⚠️ Always consult qualified professionals for serious mental health concerns
- ⚠️ Use at your own discretion

This project is intended for:
- Educational purposes
- Research in conversational AI
- Demonstrations of LLM fine-tuning techniques

## 🙏 Acknowledgments

- **Meta AI** for the LLaMA 3.2 base model
- **Unsloth AI** for the efficient fine-tuning framework
- **HuggingFace** for the Transformers library and model hosting
- **jerryjalapeno** for the NART 100k synthetic dataset
- Inspired by research in therapeutic chatbots and mental health AI

## 📖 References

- [Fine-tune LLM with QLoRA](https://dassum.medium.com/fine-tune-large-language-model-llm-on-a-custom-dataset-with-qlora-fb60abdeba07)
- [NART 100k Dataset](https://huggingface.co/datasets/jerryjalapeno/nart-100k-synthetic)
- [Unsloth Library](https://github.com/unslothai/unsloth)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [LLaMA 3.2 Model Card](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

**Note**: The base LLaMA 3.2 model is subject to Meta's license agreement. Please review [Meta's LLaMA License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE) before commercial use.

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Report bugs
- Suggest new features
- Improve documentation
- Submit pull requests

Please open an issue first to discuss major changes.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

<div align="center">

**Made with ❤️ using Meta's LLaMA 3.2 and Unsloth**

⭐ Star this repository if you found it helpful!

</div>
