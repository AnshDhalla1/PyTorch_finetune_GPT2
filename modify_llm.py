# Task 2: Model Architecture Modification Script 

import argparse
import torch
from torch import nn
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel, GPT2Config      # not a transformer library, calling pytorch implementation of GPT2
from transformers.pytorch_utils import Conv1D 


class LoRAConv1D(nn.Module):
    # LoRA for Conv1D layers (used in GPT-2)
    def __init__(self, original_layer: Conv1D, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Freeze original parameters
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # Conv1D has weight shape [out_features, in_features] (nf, nx)
        in_features = original_layer.nf   # output features (confusing naming in Conv1D)
        out_features = original_layer.nx  # input features
        
        # LoRA parameters: A with small random, B with zeros
        self.lora_A = nn.Parameter(torch.randn(rank, out_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(in_features, rank))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original Conv1D computation
        original_out = self.original_layer(x)
        
        # LoRA computation: x @ A^T @ B^T * scaling
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        
        return original_out + lora_out


class LoRALinear(nn.Module):
    # LoRA for regular Linear layers
    
    def __init__(self, original_layer: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Freeze original parameters
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # LoRA parameters
        self.lora_A = nn.Parameter(torch.randn(rank, original_layer.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(original_layer.out_features, rank))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_out = self.original_layer(x)
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return original_out + lora_out


class AdapterLayer(nn.Module):
    
    def __init__(self, hidden_size: int, adapter_size: int = 64):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, adapter_size)
        self.up_proj = nn.Linear(adapter_size, hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
        # Initialize up_proj to near-zero for stable training
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        adapted = self.down_proj(x)
        adapted = self.activation(adapted)
        adapted = self.dropout(adapted)
        adapted = self.up_proj(adapted)
        return residual + adapted


class PrefixTuning(nn.Module):
    
    def __init__(self, prefix_length: int, embedding_dim: int):
        super().__init__()
        self.prefix_embeddings = nn.Parameter(torch.randn(prefix_length, embedding_dim) * 0.02)



# Model Modification functions

def apply_lora_to_model(model: GPT2LMHeadModel, rank: int = 4, alpha: float = 1.0) -> int:
    """Apply LoRA to GPT-2 Conv1D and Linear layers."""
    target_modules = ["c_attn", "c_proj", "c_fc"]  # GPT-2 attention and MLP layers
    modified_count = 0
    
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            # get parent module
            parent_module = model
            module_path = name.split('.')
            
            for path_component in module_path[:-1]:
                parent_module = getattr(parent_module, path_component)
            
            if isinstance(module, Conv1D):
                lora_layer = LoRAConv1D(module, rank=rank, alpha=alpha)
                setattr(parent_module, module_path[-1], lora_layer)
                modified_count += 1
                print(f"   ✅ Applied LoRA to Conv1D: {name}")
                
            elif isinstance(module, nn.Linear):
                lora_layer = LoRALinear(module, rank=rank, alpha=alpha)
                setattr(parent_module, module_path[-1], lora_layer)
                modified_count += 1
                print(f" Applied LoRA to Linear: {name}")
    
    return modified_count



def apply_adapters_to_model(model: GPT2LMHeadModel, adapter_size: int = 64) -> int:
    """Apply adapter layers to transformer blocks."""
    # Freeze base model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    modified_count = 0
    
    for i, block in enumerate(model.transformer.h):  # GPT-2 uses 'h' for layers
        adapter = AdapterLayer(model.config.n_embd, adapter_size)
        
        # Store original forward method
        original_forward = block.forward
        
        def create_new_forward(orig_forward, adapter_layer, block_idx):
            def new_forward(hidden_states, **kwargs):
                outputs = orig_forward(hidden_states, **kwargs)
                
                if isinstance(outputs, tuple):
                    hidden_states = outputs[0]
                    adapted_states = adapter_layer(hidden_states)
                    new_outputs = (adapted_states,) + outputs[1:]
                    return new_outputs
                else:
                    return adapter_layer(outputs)
            return new_forward
        
        block.forward = create_new_forward(original_forward, adapter, i)
        block.adapter = adapter  # Store reference
        modified_count += 1
        print(f"  Added adapter to block {i}")
    
    return modified_count

# Apply prefix tuning to the model
def apply_prefix_tuning(model: GPT2LMHeadModel, prefix_length: int = 50) -> int:

    # Freeze all model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Add prefix embeddings
    model.prefix_embeddings = nn.Parameter(torch.randn(prefix_length, model.config.n_embd) * 0.02)
    model.prefix_embeddings.requires_grad = True
    
    print(f" Added {prefix_length} prefix embeddings")
    return 1


def apply_selective_freezing(model: GPT2LMHeadModel, freeze_layers: list = None) -> int:
    """Apply selective layer freezing strategy."""
    if freeze_layers is None:
        freeze_layers = [0, 1, 2, 3]
    
    # First freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze selected layers
    total_layers = len(model.transformer.h)
    unfrozen_layers = []
    
    for i in range(total_layers):
        if i not in freeze_layers:
            for param in model.transformer.h[i].parameters():
                param.requires_grad = True
            unfrozen_layers.append(i)
    
    # Always unfreeze the language modeling head
    for param in model.lm_head.parameters():
        param.requires_grad = True
    
    print(f"   ❄️ Frozen layers: {freeze_layers}")
    print(f"   🔥 Unfrozen layers: {unfrozen_layers}")
    
    return len(unfrozen_layers)

# To print detailed trainanility status
def print_trainable_parameters(model):
    print("\nParameter Trainability Status:")
    trainable_params = []
    frozen_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append((name, param.numel()))
            print(f"  ✅ {name}: Trainable ({param.numel():,} params)")
        else:
            frozen_params.append((name, param.numel()))
            print(f"  ❄️  {name}: Frozen ({param.numel():,} params)")
    
    trainable_count = sum(count for _, count in trainable_params)
    total_count = trainable_count + sum(count for _, count in frozen_params)
    
    print(f"\n Final Summary:")
    print(f"  Trainable: {trainable_count:,} / {total_count:,} params")
    print(f"  Ratio: {trainable_count/total_count:.4f} ({trainable_count/total_count*100:.2f}%)")



def modify_gpt2_model(modification_type: str, model: GPT2LMHeadModel, **kwargs) -> GPT2LMHeadModel:
    """Apply specified modification to the model."""
    
    print(f"\nApplying {modification_type.upper()} modification...")
    
    if modification_type == 'lora':
        rank = kwargs.get('lora_rank', 4)
        alpha = kwargs.get('lora_alpha', 1.0)
        modified_count = apply_lora_to_model(model, rank=rank, alpha=alpha)
        print(f"\nApplied LoRA with rank={rank}, alpha={alpha}")
        print(f"   Modified {modified_count} layers")
        
    elif modification_type == 'adapter':
        adapter_size = kwargs.get('adapter_size', 64)
        modified_count = apply_adapters_to_model(model, adapter_size=adapter_size)
        print(f"\n Applied Adapters with size={adapter_size}")
        print(f"   Modified {modified_count} transformer blocks")
        
    elif modification_type == 'prefix':
        prefix_length = kwargs.get('prefix_length', 50)
        modified_count = apply_prefix_tuning(model, prefix_length=prefix_length)
        print(f"\nApplied Prefix Tuning")
        
    elif modification_type == 'selective':
        freeze_layers = kwargs.get('freeze_layers', [0, 1, 2, 3])
        modified_count = apply_selective_freezing(model, freeze_layers=freeze_layers)
        print(f"\nApplied Selective Fine-tuning")
    
    return model




def parse_args():
    parser = argparse.ArgumentParser(
        description="Modify GPT-2 model with custom fine-tuning techniques",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Modification type
    parser.add_argument('--modification_type', type=str, required=True, 
                       choices=['lora', 'adapter', 'prefix', 'selective'])
    
    # Model configuration
    parser.add_argument('--vocab_size', type=int, default=50257)
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--num_attention_heads', type=int, default=12)
    parser.add_argument('--num_hidden_layers', type=int, default=12)
    parser.add_argument('--max_position_embeddings', type=int, default=1024)
    
    # Technique-specific parameters
    parser.add_argument('--lora_rank', type=int, default=4)
    parser.add_argument('--lora_alpha', type=float, default=1.0)
    parser.add_argument('--adapter_size', type=int, default=64)
    parser.add_argument('--prefix_length', type=int, default=50)
    parser.add_argument('--freeze_layers', type=str, default="0,1,2,3")
    
    return parser.parse_args()


# Main function called by training script.
def final_modify(args):
    
    print("Starting GPT-2 Model Modification...")
    
    # creating GPT-2 configuration 
    config = GPT2Config(
        vocab_size=args.vocab_size,
        n_embd=args.hidden_size,           
        n_head=args.num_attention_heads,  
        n_layer=args.num_hidden_layers,   
        n_positions=args.max_position_embeddings,  
    )
    
    # creating model with language modeling head 
    model = GPT2LMHeadModel(config)
    
    print(f"Base model created: GPT2LMHeadModel")
    print(f" Architecture: {config.n_layer} layers, {config.n_embd} hidden size")
    print(f"Output shape: [batch_size, seq_len, {config.vocab_size}]")
    
    # Parse freeze layers for selective freezing
    freeze_layers = []
    if hasattr(args, 'freeze_layers') and args.freeze_layers:
        freeze_layers = [int(x.strip()) for x in args.freeze_layers.split(',')]
    
    # Apply modification
    modified_model = modify_gpt2_model(
        modification_type=args.modification_type,
        model=model,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        adapter_size=args.adapter_size,
        prefix_length=args.prefix_length,
        freeze_layers=freeze_layers,
    )
    
    # Calculate parameter statistics
    total_params = sum(p.numel() for p in modified_model.parameters())
    trainable_params = sum(p.numel() for p in modified_model.parameters() if p.requires_grad)
    
    print(f"\n Final Parameter Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Trainable ratio: {trainable_params/total_params:.4f} ({trainable_params/total_params*100:.2f}%)")
    
    # printing detailed parameter status for verification
    print_trainable_parameters(modified_model)
    
    print(f"\nModel modification complete and ready for training!")
    return modified_model


if __name__ == "__main__":
    args = parse_args()
    final_modify(args)
