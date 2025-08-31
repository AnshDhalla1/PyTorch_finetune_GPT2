# PyTorch_finetune_GPT2


# Task 1: Alpaca Data Pipeline

The script implements a comprehensive data processing pipeline for the Stanford Alpaca dataset, specifically designed for GPT-2 style causal language model fine-tuning. The pipeline handles end-to-end data preparation from raw JSON to training-ready PyTorch tensors.

## Architecture & Data Flow

```
Raw JSON → Pandas DataFrame → GPT2 Tokenization → PyTorch Tensors → DataLoader Batches
```

## Key Features

### 🔄 Data Loading & Validation
- Loads 52,002 Alpaca instruction-response pairs from JSON format
- Validates dataset structure and file existence
- Displays dataset columns and sample entries for inspection
- Robust error handling for file I/O operations

### 🔤 Advanced Tokenization
- **GPT-2 Tokenizer**: Uses HuggingFace's pre-trained GPT2 tokenizer
- **Smart Padding**: Sets `pad_token = eos_token` for GPT-2 compatibility
- **Fixed Length Processing**: Standardizes all sequences to 512 tokens
- **Batch Processing**: Efficiently tokenizes all 52K samples at once
- **Truncation/Padding**: Handles variable-length inputs consistently

### 📊 Quality Visualization & Validation
- **Sample Inspection**: Shows original text, token IDs, and decoded text for 5 samples
- **Tokenization Verification**: Ensures reversible tokenization process
- **Debug Information**: Comprehensive logging for pipeline monitoring
- **Quality Assurance**: Validates tokenization integrity

### ⚡ Optimized PyTorch Integration
- **Custom Dataset Class**: `AlpacaDataset` wrapping tokenized data
- **Multi-worker DataLoader**: 4 parallel workers for efficient data loading
- **GPU Optimization**: Proper tensor shapes for training `[batch_size, 512]`
- **Memory Efficiency**: Shuffling enabled with consistent batch sizes

## Usage

### Basic Execution
```bash
python data_pipeline.py
```

### Modify Dataset Path
Update the `dataset_path` variable in the script:
```python
dataset_path = "/path/to/your/alpaca_```a.json"
```

## Expected Output

### 1. Dataset Loading
```
INFO:root:Dataset loaded successfully with 52002 entries.
INFO:root:Dataset columns: ['instruction', 'input', 'output']
INFO:root:First few rows:
[Sample data display...]
```

### 2. Tokenization Process
```
INFO:root:Preprocessing and tokenizing the dataset...```FO:root:Tokenization complete with shape: torch.Size([52002, 512])
```

### 3. Quality Visualization
```
INFO:root:Visualizing tokenization quality...
INFO:root:Sample 1:
INFO:root:Original Text: [Original response text]
INFO:root:Tokenized (IDs): [Token ID sequence]
INFO:root:Detokenized Text: [Reconstructed text]
```

### 4. DataLoader Creation
```
INFO:root:DataLoader created with batch size 8 and 4 workers.
INFO:root:Batch - input_ids shape: torch.Size([8, 512])
```

## Technical Specifications

### Dataset Processing
- **Total Samples**: 52,002 instruction-response pairs
- **Input Columns**: `instruction`, `input`, `output`
- **Processing Target**: Tokenizes the `output` field for response generation
- **Data Format**: JSON with records orientation

### Tokenization Configuration
- **Tokenizer**: GPT2Tokenizer from HuggingFace Transformers
- **Vocabulary Size**: 50,257 tokens
- **Max Sequence Length**: 512 tokens (fixed)
- **Padding Strategy**: Right-padding with EOS tokens
- **Truncation**: Applied to sequences exceeding max length

### DataLoader Specifications
- **Batch Size**: 8 samples per batch
- **Workers**: 4 parallel data loading processes
- **Shuffling**: Enabled for training randomization
- **Output Shape**: `torch.Size([8, 512])` per batch
- **Total Batches**: 6,501 batches per epoch (52,002 ÷ 8)

## Installation
```bash
pip install torch transformers pandas
```





# Task 2: Model Architecture Modification

This script implements custom fine-tuning techniques for GPT-2 models without relying on external PEFT libraries. All implementations are research-backed and designed for parameter-efficient training.

## Research Background & Techniques

### 1. LoRA (Low-Rank Adaptation)
**Research Foundation:** Hu et al., 2021 - "LoRA: Low-Rank Adaptation of Large Language Models"

- **Core Hypothesis:** Weight updates during adaptation have low "intrinsic rank"
- **Method:** Decomposes weight updates as ΔW = B·A where rank(B,A) << original rank
- **Implementation:** Freezes original weights, trains only low-rank matrices A and B
- **Benefits:**
  - 99%+ reduction in trainable parameters
  - No inference overhead when merged
  - Maintains performance comparable to full fine-tuning
  - Enables efficient multi-task serving

### 2. Adapters
**Research Foundation:** Houlsby et al., 2019 - "Parameter-Efficient Transfer Learning for NLP"

- **Core Insight:** Small bottleneck layers can achieve effective adaptation
- **Architecture:** Down-projection → Activation → Up-projection → Residual connection
- **Implementation:** Inserts adapter modules within each transformer block
- **Benefits:**
  - Only 2-4% additional parameters
  - Modular design for task-specific adaptation
  - Maintains high performance across diverse tasks
  - Easy to add/remove without changing base model

### 3. Prefix Tuning
**Research Foundation:** Li & Liang, 2021 - "Prefix-Tuning: Optimizing Continuous Prompts for Generation"

- **Core Concept:** Learnable "virtual tokens" prepended to input sequences
- **Method:** Optimizes continuous task-specific vectors instead of discrete tokens
- **Implementation:** Adds trainable prefix embeddings to model input
- **Benefits:**
  - Extremely parameter efficient (<0.1% of model parameters)
  - Strong performance on generation tasks
  - Enables rapid task switching by swapping prefixes
  - No architectural changes required

### 4. Selective Layer Freezing
**Research Foundation:** Strategic layer-wise fine-tuning approach

- **Strategy:** Freezes early transformer layers, trains upper layers
- **Rationale:** Early layers capture general features, later layers capture task-specific patterns
- **Implementation:** Configurable layer freezing with trainable output head
- **Benefits:**
  - Significant reduction in training cost
  - Preserves low-level learned representations
  - Maintains performance for similar-domain tasks


## Usage Examples

### LoRA Fine-tuning
```bash
# Basic LoRA with rank 4
python modify_llm.py --modification_type lora --lora_rank 4 --lora_alpha 1.0

# High-capacity LoRA for complex tasks
python modify_llm.py --modification_type lora --lora_rank 16 --lora_alpha 32.0

```

### Adapter Fine-tuning
```bash
# Standard adapter configuration
python modify_llm.py --modification_type adapter --adapter_size 64

# Large adapters for complex tasks
python modify_llm.py --modification_type adapter --adapter_size 256

# Minimal adapters for efficiency
python modify_llm.py --modification_type adapter --adapter_size 32
```

### Prefix Tuning
```bash
# Standard prefix tuning
python modify_llm.py --modification_type prefix --prefix_length 50

# Short prefix for simple tasks
python modify_llm.py --modification_type prefix --prefix_length 20

```

### Selective Freezing
```bash
# Freeze first 4 layers
python modify_llm.py --modification_type selective --freeze_layers "0,1,2,3"

# Freeze first 6 layers
python modify_llm.py --modification_type selective --freeze_layers "0,1,2,3,4,5"

# Train only last 2 layers
python modify_llm.py --modification_type selective --freeze_layers "0,1,2,3,4,5,6,7,8,9"
```

## Technical Specifications

### Model Configuration
- **Base Architecture:** GPT-2 (12 layers, 768 hidden size)
- **Vocabulary Size:** 50,257 tokens
- **Attention Heads:** 12
- **Max Position Embeddings:** 1024
- **Target Modules:** c_attn, c_proj, c_fc (attention and MLP layers)

### Parameter Efficiency Comparison
| Technique | Trainable Parameters | Typical Performance | Memory Overhead |
|-----------|---------------------|-------------------|-----------------|
| **Full Fine-tuning** | 100% | Baseline | High |
| **LoRA (r=4)** | ~0.1% | 95-100% of full FT | Minimal |
| **Adapters (64)** | ~2-4% | 90-98% of full FT | Low |
| **Prefix Tuning (50)** | <0.1% | 85-95% of full FT | Minimal |
| **Selective (6 frozen)** | ~50% | 85-95% of full FT | Medium |

### Output Information
The script provides detailed parameter analysis:
```
📊 Final Parameter Statistics:
   Total parameters: 124,439,808
   Trainable parameters: 294,912
   Trainable ratio: 0.0024 (0.24%)

🔍 Parameter Trainability Status:
  ✅ transformer.h.0.c_attn.lora_A: Trainable (3,072 params)
  ✅ transformer.h.0.c_attn.lora_B: Trainable (9,216 params)
  ❄️  transformer.h.0.c_attn.original_layer.weight: Frozen```,769,472 params)
  ...
```

## Implementation Features

### Architecture Handling
- **Conv1D Support:** Proper handling of GPT-2's Conv1D layers
- **Layer Navigation:** Dynamic module replacement and parent traversal
- **State Preservation:** Maintains original model structure and weights

### Parameter Management
- **Freeze Control:** Selective parameter freezing with detailed tracking
- **Memory Efficiency:** Only trainable parameters consume gradient memory
- **Statistics:** Comprehensive parameter counting and ratio calculation

- **Error Handling:** Graceful handling of unsupported layer types
- **Validation:** Parameter shape validation and compatibility checks
- **Modularity:** Clean separation of different techniques

