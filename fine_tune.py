#!/usr/bin/env python
import os
import json
import argparse
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

def load_dataset(data_path):
    """
    Loads a dataset from a JSON file and converts it to HuggingFace Dataset format.
    
    Args:
        data_path: Path to the JSON file with data
        
    Returns:
        Dataset for training and validation
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts = []
    for item in data:
        function_code = item.get('function_code', '')
        test_code = item.get('test_code', '')
        
        prompt = f"""Generate a test for the following Python function:

```python
{function_code}
```

The test should verify that the function works correctly, including edge cases.
"""
        
        response = f"""```python
{test_code}
```"""
        
        full_text = f"{prompt}\n\n{response}"
        texts.append({"text": full_text})
    
    dataset = Dataset.from_list(texts)
    dataset = dataset.train_test_split(test_size=0.1)
    
    return dataset

def tokenize_function(examples, tokenizer, max_length):
    """
    Tokenizes text examples for training.
    
    Args:
        examples: Examples to tokenize
        tokenizer: Tokenizer
        max_length: Maximum sequence length
        
    Returns:
        Tokenized examples
    """
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model for Python test generation")
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-1b", 
                       help="Name or path of the base model to fine-tune")
    parser.add_argument("--data_path", type=str, default="python_code_test_dataset.json",
                       help="Path to the JSON dataset")
    parser.add_argument("--output_dir", type=str, default="./fine_tuned_model",
                       help="Directory to save the fine-tuned model")
    parser.add_argument("--max_length", type=int, default=1024,
                       help="Maximum sequence length for tokenization")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate for fine-tuning")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f"Loading model {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True
    )
    
    print("Loading and preparing dataset...")
    dataset = load_dataset(args.data_path)
    
    tokenized_dataset = {
        "train": dataset["train"].map(
            lambda examples: tokenize_function(examples, tokenizer, args.max_length),
            batched=True,
            remove_columns=["text"]
        ),
        "test": dataset["test"].map(
            lambda examples: tokenize_function(examples, tokenizer, args.max_length),
            batched=True,
            remove_columns=["text"]
        )
    }
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_steps=500,
        save_steps=1000,
        warmup_steps=500,
        learning_rate=args.learning_rate,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=100,
        eval_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print("Fine-tuning complete!")

if __name__ == "__main__":
    main() 