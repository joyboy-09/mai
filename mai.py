import gc
import torch
import os
import matplotlib.pyplot as plt
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
from rich.progress import Progress, TaskID
from rich.console import Console
import numpy as np
import psutil

# Memory clearing function
def clear_memory():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        memory = psutil.virtual_memory()
        print(f"System Memory - Available: {memory.available / 1024**3:.1f}GB / Total: {memory.total / 1024**3:.1f}GB")

clear_memory()

# Configuration
MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"
DATASET_NAME = "mounikaiiith/Telugu_Emotion"
OUTPUT_DIR = "./phi35_telugu_optimized"
BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 2e-5
MAX_LENGTH = 256
GRADIENT_ACCUMULATION = 4

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/logs", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/plots", exist_ok=True)

console = Console()

# Memory monitoring function
def print_memory_usage(stage=""):
    memory = psutil.virtual_memory()
    process = psutil.Process(os.getpid())
    process_memory = process.memory_info().rss / 1024**3
    console.print(f"[bold blue]{stage}[/bold blue] - Process Memory: {process_memory:.2f}GB, System Available: {memory.available / 1024**3:.2f}GB")

# Load tokenizer
console.print("[bold green]Loading tokenizer and model...[/bold green]")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print_memory_usage("After tokenizer")

# Device setup
device = "mps" if torch.backends.mps.is_available() else "cpu"
console.print(f"[bold cyan]Using device: {device}[/bold cyan]")

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map=None,
    use_cache=False,
    low_cpu_mem_usage=True,
)

model = model.to(device)
print_memory_usage("After model loading")

# Prepare model for training
model.train()
model.enable_input_require_grads()

# LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["qkv_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print_memory_usage("After LoRA setup")

# Load and preprocess dataset
console.print("[bold green]Loading and preprocessing dataset...[/bold green]")
ds = load_dataset(DATASET_NAME, split="train")
console.print(f"Dataset size: {len(ds)} examples")

def tokenize_fn(batch):
    return tokenizer(
        batch["Sentence"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_special_tokens_mask=True
    )

train_test_split = ds.train_test_split(test_size=0.1, seed=42)
tokenized_train = train_test_split['train'].map(tokenize_fn, batched=True, remove_columns=ds.column_names)
tokenized_eval = train_test_split['test'].map(tokenize_fn, batched=True, remove_columns=ds.column_names)

console.print(f"Training examples: {len(tokenized_train)}")
console.print(f"Validation examples: {len(tokenized_eval)}")

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# FIXED: Enhanced Trainer class
class EnhancedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # FIXED: Correct __init__ call
        self.train_losses = []
        self.eval_losses = []
        self.learning_rates = []
        self.step_times = []
        self.start_time = datetime.now()
    
    def log(self, logs, *args, **kwargs):
        super().log(logs, *args, **kwargs)
        
        # Collect metrics for plotting
        if "train_loss" in logs:
            self.train_losses.append(logs["train_loss"])
        if "eval_loss" in logs:
            self.eval_losses.append(logs["eval_loss"])
        if "learning_rate" in logs:
            self.learning_rates.append(logs["learning_rate"])
        
        # Print enhanced progress
        current_time = datetime.now()
        elapsed = (current_time - self.start_time).total_seconds() / 3600
        if logs.get("epoch"):
            console.print(f"[bold yellow]Epoch {logs['epoch']:.2f}[/bold yellow] | "
                         f"Loss: {logs.get('train_loss', 0):.4f} | "
                         f"LR: {logs.get('learning_rate', 0):.2e} | "
                         f"Time: {elapsed:.2f}h")
        
        # Plot loss every 10 steps
        if len(self.train_losses) % 10 == 0 and len(self.train_losses) > 0:
            self.plot_training_progress()
    
    def plot_training_progress(self):
        plt.figure(figsize=(15, 5))
        
        # Loss plot
        plt.subplot(1, 3, 1)
        if self.train_losses:
            plt.plot(self.train_losses, label='Train Loss', color='blue')
        if self.eval_losses:
            plt.plot(self.eval_losses, label='Eval Loss', color='red')
        plt.title('Training Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Learning rate plot
        plt.subplot(1, 3, 2)
        if self.learning_rates:
            plt.plot(self.learning_rates, color='green')
        plt.title('Learning Rate')
        plt.xlabel('Steps')
        plt.ylabel('LR')
        plt.grid(True)
        
        # Memory usage plot
        plt.subplot(1, 3, 3)
        memory = psutil.virtual_memory()
        process = psutil.Process(os.getpid())
        process_memory = process.memory_info().rss / 1024**3
        available_memory = memory.available / 1024**3
        
        plt.bar(['Used', 'Available'],
               [process_memory, available_memory],
               color=['red', 'green'])
        plt.title('System Memory (GB)')
        plt.ylabel('Memory (GB)')
        
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/plots/training_progress.png", dpi=100, bbox_inches='tight')
        plt.close()

# FIXED: Training arguments with fp16 explicitly disabled
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    warmup_steps=100,
    
    # Logging and saving
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=10,
    logging_first_step=True,
    save_steps=200,
    save_total_limit=3,
    
    # Evaluation settings
    eval_strategy="steps",
    eval_steps=200,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    
    # CRITICAL: Explicitly disable mixed precision for MPS
    fp16=False,  # Must be False for Apple Silicon
    bf16=False,  # Must be False for Apple Silicon
    
    # Performance optimizations
    gradient_checkpointing=True,
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
    group_by_length=True,
    
    # Optimizer settings
    max_grad_norm=1.0,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    weight_decay=0.01,
    report_to="tensorboard",
    remove_unused_columns=False,
)

# Verify settings before creating trainer
print("=== VERIFICATION ===")
print(f"fp16: {training_args.fp16}")  # Should be False
print(f"bf16: {training_args.bf16}")  # Should be False
print(f"Device: {model.device}")      # Should show mps:0

# Initialize trainer
trainer = EnhancedTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=data_collator,
)

# Start training
console.print("[bold green]Starting enhanced training...[/bold green]")
console.print(f"[bold cyan]Expected Memory usage: ~12-14GB[/bold cyan]")
console.print(f"[bold cyan]Training on Apple M4 with MPS acceleration[/bold cyan]")

start_time = datetime.now()

# Clear memory before training
clear_memory()

# Check current memory usage
memory = psutil.virtual_memory()
process = psutil.Process(os.getpid())
print(f"Process Memory: {process.memory_info().rss / 1024**3:.2f} GB")
print(f"Available Memory: {memory.available / 1024**3:.2f} GB")

# Train the model
trainer.train()

end_time = datetime.now()
total_time = (end_time - start_time).total_seconds() / 3600

console.print(f"[bold green]Training completed in {total_time:.2f} hours![/bold green]")

# Save final model
trainer.save_model()
console.print(f"[bold green]Model saved to {OUTPUT_DIR}[/bold green]")
