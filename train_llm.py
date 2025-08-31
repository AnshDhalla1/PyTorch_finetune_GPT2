"""
Task 3: Training Integration Pipeline 
"""

import argparse
import os
import sys
import json
import time
import math
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import torch.nn.functional as F

from data_pipeline import final_dataset_loader
from modify_llm import final_modify


# Logging
def setup_logging(log_dir: str, log_level: str = "INFO"):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

# metrics tracker for training
class MetricsTracker:
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.train_losses = []
        self.eval_losses = []
        self.learning_rates = []
        self.perplexities = []
        self.step_times = []
        self.epoch_times = []
        self.gradient_norms = []
        
    def update_train_step(self, loss: float, lr: float, step_time: float, grad_norm: float = None):
        self.train_losses.append(loss)
        self.learning_rates.append(lr)
        self.step_times.append(step_time)
        self.perplexities.append(math.exp(min(loss, 10)))  # Clamp to avoid overflow
        if grad_norm is not None:
            self.gradient_norms.append(grad_norm)
    
    def update_eval(self, eval_loss: float):
        self.eval_losses.append(eval_loss)
    
    def get_averages(self, window: int = 100) -> Dict[str, float]:
        """Get recent averages for monitoring."""
        if not self.train_losses:
            return {}
        
        recent_losses = self.train_losses[-window:]
        recent_times = self.step_times[-window:]
        recent_perplexities = self.perplexities[-window:]
        
        return {
            'avg_train_loss': sum(recent_losses) / len(recent_losses),
            'avg_perplexity': sum(recent_perplexities) / len(recent_perplexities),
            'avg_step_time': sum(recent_times) / len(recent_times),
            'current_lr': self.learning_rates[-1] if self.learning_rates else 0,
            'latest_eval_loss': self.eval_losses[-1] if self.eval_losses else None
        }
    
    def save_metrics(self, save_path: str):
        # Save metrics to JSON file
        metrics_data = {
            'train_losses': self.train_losses,
            'eval_losses': self.eval_losses,
            'learning_rates': self.learning_rates,
            'perplexities': self.perplexities,
            'step_times': self.step_times,
            'epoch_times': self.epoch_times,
            'gradient_norms': self.gradient_norms
        }
        
        with open(save_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)

# counting total and trainable parameters
def count_parameters(model: nn.Module) -> Dict[str, int]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'trainable_ratio': trainable_params / total_params,
        'frozen_params': total_params - trainable_params
    }


# Save comprehensive training checkpoint
def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                   scheduler, epoch: int, step: int, loss: float, 
                   metrics: MetricsTracker, checkpoint_dir: str, 
                   is_best: bool = False) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'loss': loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': {
            'train_losses': metrics.train_losses,
            'eval_losses': metrics.eval_losses,
            'learning_rates': metrics.learning_rates
        },
        'timestamp': datetime.now().isoformat(),
        'model_config': {
            'total_params': count_parameters(model)['total_params'],
            'trainable_params': count_parameters(model)['trainable_params']
        }
    }
    
    # save best checkpoint
    if is_best:
        best_path = checkpoint_dir / "best_checkpoint.pt"
        torch.save(checkpoint, best_path)
        logging.getLogger(__name__).info(f"💎 New best checkpoint saved: {best_path}")
    
    return checkpoint_path


def create_optimizer_and_scheduler(model: nn.Module, args):
    # only including trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # creating AdamW optimizer 
    optimizer = AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=args.eps
    )
    
    #  learning rate scheduler
    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.total_steps, eta_min=args.min_lr)
    elif args.scheduler == "linear":
        scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_steps)
    elif args.scheduler == "warmup_cosine":
        # Warmup followed by cosine decay
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_steps)
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=args.total_steps - args.warmup_steps, eta_min=args.min_lr
        )
        scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], [args.warmup_steps])
    else:
        scheduler = None
    
    return optimizer, scheduler



# Train and Eval Functions
def train_step(model: nn.Module, batch: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer, 
              device: torch.device, gradient_accumulation_steps: int = 1, 
              max_grad_norm: float = 1.0) -> Tuple[float, float]:
    # Single training step with gradient accumulation and clipping
    
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    
    # Forward pass
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    loss = outputs.loss
    
    # Scale loss for gradient accumulation
    loss = loss / gradient_accumulation_steps
    
    # Backward pass
    loss.backward()
    
    # Calculate gradient norm before clipping
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    
    return loss.item() * gradient_accumulation_steps, grad_norm.item()


# Model Evaluation
def evaluate_model(model: nn.Module, eval_dataloader: DataLoader, 
                  device: torch.device, logger) -> Dict[str, float]:
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            
            # Calculate number of tokens 
            num_tokens = attention_mask.sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(min(avg_loss, 10))  # Clamp to avoid overflow
    
    results = {
        'eval_loss': avg_loss,
        'eval_perplexity': perplexity,
        'eval_tokens': total_tokens
    }
    
    logger.info(f"Evaluation Results:")
    logger.info(f"   Loss: {avg_loss:.4f}")
    logger.info(f"   Perplexity: {perplexity:.2f}")
    logger.info(f"   Tokens: {total_tokens:,}")
    
    return results


# training for one epoch with monitoring
def train_epoch(model: nn.Module, train_dataloader: DataLoader, 
               eval_dataloader: DataLoader, optimizer: torch.optim.Optimizer, 
               scheduler, device: torch.device, epoch: int, args, 
               logger, metrics: MetricsTracker) -> float:
    
    model.train()
    epoch_start_time = time.time()
    step_count = 0
    accumulated_loss = 0
    
    for batch_idx, batch in enumerate(train_dataloader):
        step_start_time = time.time()
        
        # Training step
        loss, grad_norm = train_step(
            model, batch, optimizer, device,
            args.gradient_accumulation_steps, args.max_grad_norm
        )
        
        accumulated_loss += loss
        step_count += 1
        
        # optimizer step 
        if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            
            if scheduler:
                scheduler.step()
        
        # update metrics
        step_time = time.time() - step_start_time
        current_lr = optimizer.param_groups[0]['lr']
        metrics.update_train_step(loss, current_lr, step_time, grad_norm)
        
        # logging
        if batch_idx % args.log_interval == 0:
            recent_metrics = metrics.get_averages(window=50)
            logger.info(
                f"Epoch {epoch} | Step {batch_idx:4d}/{len(train_dataloader)} | "
                f"Loss: {recent_metrics['avg_train_loss']:.4f} | "
                f"PPL: {recent_metrics['avg_perplexity']:.2f} | "
                f"LR: {current_lr:.2e} | "
                f"Grad: {grad_norm:.3f} | "
                f"Time: {step_time:.3f}s"
            )
        
        # early stopping for testing
        if args.max_steps and step_count >= args.max_steps:
            logger.info(f"Reached max_steps ({args.max_steps}), stopping epoch early")
            break
    
    # Epoch evaluation
    if eval_dataloader and epoch % args.eval_interval == 0:
        eval_results = evaluate_model(model, eval_dataloader, device, logger)
        metrics.update_eval(eval_results['eval_loss'])
    
    epoch_time = time.time() - epoch_start_time
    metrics.epoch_times.append(epoch_time)
    
    avg_epoch_loss = accumulated_loss / step_count
    logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s | Avg Loss: {avg_epoch_loss:.4f}")
    
    return avg_epoch_loss


# Main training loop with full integration
def main_training_loop(args, logger):
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    logger.info(f"🔧 Using device: {device}")
    
    # Create modified model from Task 2
    logger.info(" Creating model with Task 2 modifications -")
    model = final_modify(args)
    model.to(device)
    
    # Log parameter information
    param_stats = count_parameters(model)
    logger.info(f"Model Parameters:")
    logger.info(f"   Total: {param_stats['total_params']:,}")
    logger.info(f"   Trainable: {param_stats['trainable_params']:,}")
    logger.info(f"   Frozen: {param_stats['frozen_params']:,}")
    logger.info(f"   Trainable Ratio: {param_stats['trainable_ratio']:.4f} ({param_stats['trainable_ratio']*100:.2f}%)")
    
    # load data from Task 1
    logger.info(" Loading dataset using Task 1 pipeline...")
    train_dataloader = final_dataset_loader(args.dataset_path)
    
    # creating evaluation dataloader 
    eval_dataloader = None
    if args.eval_dataset_path:
        eval_dataloader = final_dataset_loader(args.eval_dataset_path)
    elif hasattr(train_dataloader.dataset, '__len__') and len(train_dataloader.dataset) > 100:
        
        # Using portion of training data for evaluation
        logger.info(" Using portion of training data for evaluation")
        eval_dataloader = train_dataloader  # Simplified for demo
    
    logger.info(f"Training batches: {len(train_dataloader)}")
    if eval_dataloader:
        logger.info(f"Evaluation batches: {len(eval_dataloader)}")
    
    # calculate total steps for scheduler
    args.total_steps = args.epochs * len(train_dataloader) // args.gradient_accumulation_steps
    
    # creating optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(model, args)
    logger.info(f"Optimizer: {type(optimizer).__name__}")
    logger.info(f"Scheduler: {type(scheduler).__name__ if scheduler else 'None'}")
    logger.info(f"Total training steps: {args.total_steps}")
    
    # initializing metrics tracking
    metrics = MetricsTracker()
    best_eval_loss = float('inf')
    
    # Training loop
    logger.info("Starting training...")
    logger.info("=" * 80)
    
    for epoch in range(args.epochs):
        try:
            avg_loss = train_epoch(
                model, train_dataloader, eval_dataloader, optimizer, scheduler,
                device, epoch, args, logger, metrics
            )
            
            # Checkpointing
            is_best = False
            if eval_dataloader and metrics.eval_losses:
                current_eval_loss = metrics.eval_losses[-1]
                is_best = current_eval_loss < best_eval_loss
                if is_best:
                    best_eval_loss = current_eval_loss
            
            # Save checkpoint
            checkpoint_path = save_checkpoint(
                model, optimizer, scheduler, epoch, 
                epoch * len(train_dataloader), avg_loss,
                metrics, args.output_dir, is_best=is_best
            )
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Save metrics
            metrics_path = Path(args.output_dir) / f"metrics_epoch_{epoch}.json"
            metrics.save_metrics(metrics_path)
            
        except KeyboardInterrupt:
            logger.info("⚠️ Training interrupted by user")
            break
        except Exception as e:
            logger.error(f"❌ Error during epoch {epoch}: {str(e)}")
            raise
    
    # Final evaluation
    if eval_dataloader:
        logger.info("🏁 Final evaluation...")
        final_results = evaluate_model(model, eval_dataloader, device, logger)
        logger.info(f"🎉 Final Results: Loss={final_results['eval_loss']:.4f}, PPL={final_results['eval_perplexity']:.2f}")
    
    # Save final metrics
    final_metrics_path = Path(args.output_dir) / "final_metrics.json"
    metrics.save_metrics(final_metrics_path)
    logger.info(f"Final metrics saved: {final_metrics_path}")
    
    logger.info(" Training completed successfully!")
    return model, metrics

#CLI Commands

def parse_args():
    parser = argparse.ArgumentParser(
        description="Task 3: Training Integration Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments: Task 1 integration
    parser.add_argument('--dataset_path', type=str, required=True, 
                       help="Path to training dataset (Task 1)")
    parser.add_argument('--eval_dataset_path', type=str, 
                       help="Path to evaluation dataset (optional)")
    
    # Model arguments: Task 2 integration
    parser.add_argument('--modification_type', type=str, required=True,
                       choices=['lora', 'adapter', 'prefix', 'selective', 'classification'],
                       help="Model modification type (Task 2)")
    parser.add_argument('--vocab_size', type=int, default=50257)
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--num_attention_heads', type=int, default=12)
    parser.add_argument('--num_hidden_layers', type=int, default=12)
    parser.add_argument('--max_position_embeddings', type=int, default=1024)
    
    # technique parameters
    parser.add_argument('--lora_rank', type=int, default=4, help="LoRA rank")
    parser.add_argument('--lora_alpha', type=float, default=8.0, help="LoRA alpha")
    parser.add_argument('--adapter_size', type=int, default=64, help="Adapter size")
    parser.add_argument('--prefix_length', type=int, default=50, help="Prefix length")
    parser.add_argument('--freeze_layers', type=str, default="0,1,2,3", help="Layers to freeze (selective)")
    
    # training configuration
    parser.add_argument('--epochs', type=int, default=3, help="Number of epochs")
    parser.add_argument('--learning_rate', type=float, default=1e-5, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay")
    parser.add_argument('--beta1', type=float, default=0.9, help="Adam beta1")
    parser.add_argument('--beta2', type=float, default=0.999, help="Adam beta2")
    parser.add_argument('--eps', type=float, default=1e-8, help="Adam epsilon")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Gradient accumulation")
    
    # scheduler configuration
    parser.add_argument('--scheduler', type=str, default='warmup_cosine',
                       choices=['cosine', 'linear', 'warmup_cosine', 'none'],
                       help="Learning rate scheduler")
    parser.add_argument('--warmup_steps', type=int, default=100, help="Warmup steps")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="Minimum learning rate")
    
    # monitoring and logging
    parser.add_argument('--log_interval', type=int, default=10, help="Steps between logging")
    parser.add_argument('--eval_interval', type=int, default=1, help="Epochs between evaluation")
    parser.add_argument('--output_dir', type=str, default='./outputs', help="Output directory")
    parser.add_argument('--log_level', type=str, default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    # other options
    parser.add_argument('--max_steps', type=int, help="Maximum training steps (for testing)")
    parser.add_argument('--no_cuda', action='store_true', help="Disable CUDA")
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(args.output_dir, args.log_level)
    
    # Log configuration
    logger.info("TASK 3: Training Integration Pipeline")
    logger.info("=" * 60)
    logger.info("Configuration:")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"  {arg}: {value}")
    logger.info("=" * 60)
    
    # Start training
    try:
        model, metrics = main_training_loop(args, logger)
        logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise




# def train(model, dataloader, epochs=3, learning_rate=1e-5, max_grad_norm=1.0):
#     """
#     Backwards compatible training function (simplified version of your original).
#     For production use, prefer the main_training_loop function above.
#     """
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     loss_fn = torch.nn.CrossEntropyLoss()
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)
#     model.train()
    
#     for epoch in range(epochs):
#         total_loss = 0
#         for batch in dataloader:
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
            
#             optimizer.zero_grad()
#             outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
#             loss = outputs.loss
#             loss.backward()
            
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
#             optimizer.step()
#             total_loss += loss.item()
        
#         scheduler.step()
#         current_lr = scheduler.get_last_lr()[0]
#         print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}, LR: {current_lr:.6f}")


# def main():
#     """Backwards compatible main function."""
#     args = parse_args()
    
#     # Create DataLoader for dataset loading and preprocessing
#     dataloader = final_dataset_loader(args.dataset_path)
    
#     # Modify the GPT-2 model based on user input
#     modified_model = final_modify(args)
    
#     # Train the model (using simple version for backwards compatibility)
#     train(modified_model, dataloader, epochs=args.epochs, learning_rate=args.learning_rate)
