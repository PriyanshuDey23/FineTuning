# Medical Reasoning Fine-Tuning with DeepSeek-R1

![](./Final/finetuning%20graph%20.png)
![](./Final/train%20summary.png)

A comprehensive project for fine-tuning the DeepSeek-R1-Distill-Llama-8B model on medical reasoning tasks using Low-Rank Adaptation (LoRA) and the Unsloth framework. This implementation enables the model to perform advanced clinical reasoning with chain-of-thought explanations.

## 🚀 Features

- **Advanced Medical Reasoning**: Fine-tuned on the FreedomIntelligence medical-o1-reasoning-SFT dataset
- **Efficient Fine-Tuning**: Uses LoRA adapters for parameter-efficient training
- **Chain-of-Thought Responses**: Generates structured medical answers with reasoning traces
- **Quantized Training**: 4-bit quantization for memory efficiency
- **Experiment Tracking**: Integrated with Weights & Biases (WandB) for monitoring
- **Hugging Face Integration**: Model hosted and accessible via Hugging Face Hub
- **Inference Scripts**: Ready-to-use scripts for testing and deployment

## 📋 Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (recommended: 16GB+ VRAM)
- **RAM**: 16GB+ system RAM
- **Storage**: 20GB+ free space for model and datasets

### Software
- **Python**: 3.8 or higher
- **CUDA**: 11.8 or higher (for GPU acceleration)
- **Git**: For cloning repositories

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/PriyanshuDey23/FineTuning.git
cd medical-reasoning-finetuning
```

### 2. Create Virtual Environment
```bash
# Using conda (recommended)
conda create -n medical-finetune python=3.10
conda activate medical-finetune

# Or using venv
python -m venv medical-finetune
source medical-finetune/bin/activate  # On Windows: medical-finetune\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install unsloth
pip install transformers datasets trl wandb huggingface-hub
```

### 4. Setup API Tokens

#### Hugging Face Token
1. Create an account at [Hugging Face](https://huggingface.co)
2. Generate a token with write permissions
3. Set environment variable:
```bash
export HF_TOKEN="your_huggingface_token"
```

#### Weights & Biases Token (Optional)
```bash
export WANDB_API_KEY="your_wandb_api_key"
```

## 📊 Dataset

This project uses the [FreedomIntelligence/medical-o1-reasoning-SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT) dataset, which contains:
- Medical questions requiring clinical reasoning
- Chain-of-thought explanations
- Evidence-based medical responses
- Training split with 500 samples used for fine-tuning

## 🏋️ Training

### Quick Start Training
```bash
python Final/Doctor_Finetuned_saved.py
```

### Training Configuration
- **Model**: DeepSeek-R1-Distill-Llama-8B
- **Method**: LoRA fine-tuning
- **Batch Size**: 2 (with gradient accumulation of 4)
- **Learning Rate**: 2e-4
- **Epochs**: 1 (60 steps)
- **Max Sequence Length**: 2048 tokens

### Key Training Parameters
```python
# LoRA Configuration
r = 16
lora_alpha = 16
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

## 🧪 Testing & Inference

### Load Fine-Tuned Model
```python
from unsloth import FastLanguageModel

# Load base model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    load_in_4bit=True
)

# Load LoRA adapter
model.load_adapter("babai2003/deepseek-r1-medical-lora")
FastLanguageModel.for_inference(model)
```

### Run Inference
```bash
python Final/finetune_model_testing.py
```

### Example Usage
```python
question = """A 61-year-old woman with involuntary urine leakage during coughing..."""

inputs = tokenizer([prompt_style.format(question=question)], return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=1200)
response = tokenizer.batch_decode(outputs)[0]
```

## 📁 Project Structure

```
FineTuning/
├── README.md
├── LICENSE
├── doctor.py
├── Doctor (1).ipynb          # Initial exploration notebook
├── Doctor (2).ipynb          # Model setup notebook
├── Doctor (3).ipynb          # Training notebook
└── Final/
    ├── Doctor_Finetuned_saved.py     # Complete training script
    ├── Doctor_Finetuned_saved.ipynb  # Training notebook
    ├── finetune_model_testing.py     # Inference testing script
    ├── Finetune_model_testing.ipynb  # Testing notebook
    └── doctor.py                     # Utility functions
```

## 🔬 Model Performance

The fine-tuned model demonstrates improved performance on medical reasoning tasks:

- **Structured Responses**: Generates clinically appropriate answers
- **Chain-of-Thought**: Provides step-by-step reasoning
- **Evidence-Based**: Incorporates medical knowledge and guidelines
- **Differential Diagnosis**: Considers multiple clinical possibilities

### Before vs After Fine-Tuning
- **Pre-trained**: Generic responses without medical context
- **Fine-tuned**: Specialized medical reasoning with clinical accuracy



### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to functions
- Test changes thoroughly
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **DeepSeek Team**: For the R1-Distill-Llama-8B model
- **FreedomIntelligence**: For the medical reasoning dataset
- **Unsloth Team**: For the efficient fine-tuning framework
- **Hugging Face**: For model hosting and transformers library



## 🔗 Links

- [Hugging Face Model](https://huggingface.co/babai2003/deepseek-r1-medical-lora)
- [Dataset](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)
- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [DeepSeek Model](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B)

---

*This project is for educational and research purposes. Always consult qualified medical professionals for clinical decisions.*