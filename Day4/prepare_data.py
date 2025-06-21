# prepare_data.py
import argparse
import os
from transformers import AutoTokenizer
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Pre-tokenize a dataset for language model training.")
    parser.add_argument("--tokenizer_name", type=str, required=True, help="Name or path of the tokenizer.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to process.")
    parser.add_argument("--output_path", type=str, required=True, help="Path on EBS to save the tokenized dataset.")
    parser.add_argument("--sequence_length", type=int, default=2048, help="The sequence length for tokenization.")
    parser.add_argument("--num_proc", type=int, default=8, help="Number of CPU cores to use for tokenization.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading dataset '{args.dataset_name}'...")
    # Load the raw dataset (not streaming)
    dataset = load_dataset(args.dataset_name, split="train")

    print(f"Loading tokenizer '{args.tokenizer_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)

    if tokenizer.pad_token is None:
        # Many models don't have a pad token, so we use the end-of-sequence token instead
        tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer `pad_token` set to `eos_token`.")

    def tokenize_function(examples):
        # We perform truncation but NOT padding. Padding will be handled dynamically during training.
        # This saves a massive amount of disk space.
        return tokenizer(examples['text'], truncation=True, max_length=args.sequence_length)

    print(f"Tokenizing dataset with {args.num_proc} processes...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=args.num_proc,
        remove_columns=dataset.column_names # Remove all original columns
    )

    print(f"Saving tokenized dataset to '{args.output_path}'...")
    tokenized_dataset.save_to_disk(args.output_path)
    
    print(f"\n🎉 Dataset preparation complete!")
    print(f"Tokenized dataset saved at: {args.output_path}")

if __name__ == "__main__":
    main()
    
    
"""
# Example usage:
python prepare_data.py \
    --tokenizer_name "HuggingFaceTB/SmolLM2-1.7B" \
    --dataset_name "HuggingFaceTB/cosmopedia-100k" \
    --output_path "/home/ubuntu/data/final_training/Day4/New_Approach/cosmopedia-100k/cosmopedia-100k-tokenized" \
    --sequence_length 2048 \
    --num_proc 16 # Use a good number of CPU cores
"""