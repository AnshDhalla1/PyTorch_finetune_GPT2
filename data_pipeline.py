import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

'''
# Flow visualisation for the Data Pipeline
Raw JSON → Pandas DataFrame → GPT2 Tokenization → PyTorch Tensors → DataLoader Batches
'''

# Loading dataset
def load_dataset(dataset_path):
   
   # Loads Alpaca json dataset & returns pd.dataframe which is the loaded dataset
    if not os.path.exists(dataset_path):
        logging.error(f"Dataset path does not exist: {dataset_path}")
        return None
    
    try:
        dataset = pd.read_json(dataset_path, orient='records')  # Assume records orientation
        logging.info(f"Dataset loaded successfully with {len(dataset)} entries.")
        
        # Printing the columns and first few rows of the dataset to check its structure
        logging.info(f"Dataset columns: {dataset.columns.tolist()}")
        logging.info(f"First few rows: \n{dataset.head()}")
        
        return dataset
    
    except Exception as e:
        logging.error(f"Error loading dataset: {str(e)}")
        return None

# Data Preprocessing and Tokenization
def preprocess_and_tokenize(dataset, tokenizer, max_length=512):
    
    # Tokenizing output field for each entry in dataset
    encodings = tokenizer(
        dataset['output'].tolist(),  # Using output field for tokenization
        truncation=True, 
        padding=True,
        max_length=max_length,  # Limit to max length
        return_tensors='pt'     # Return PyTorch tensors
    )
    logging.info(f"Tokenization complete with shape: {encodings['input_ids'].shape}")
    
    return encodings

# PyTorch Dataset
class AlpacaDataset(Dataset):
    def __init__(self, tokenized_data):
        # PyTorch dataset class for Alpaca data
        # arg - tokenized input data (from GPT2 tokenizer)
        self.input_ids = tokenized_data['input_ids']
        self.attention_mask = tokenized_data['attention_mask']
    
    def __len__(self):
        return len(self.input_ids)      #return total sample no
    
    def __getitem__(self, idx):
        # fetch sample from dataset. idx = index of sample. 
        # returns sample dict of (input_ids + attn_mask)
        sample = {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx]
        }
        return sample

# DataLoader for batching and shuffling
def create_dataloader(dataset, batch_size=8, num_workers=4):
  
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,              # size of each batch
        shuffle=True,   
        num_workers=num_workers             # no of workers for loading data
    )
    logging.info(f"DataLoader created with batch size {batch_size} and {num_workers} workers.")
    
    return dataloader           # returning Pytorch dataloader instance


def visualize_tokenization_quality(dataset, tokenizer, num_samples=5):
   
    #Visualize tokenization process for few sample - Validation
    # num_samples = no of samples to visualize
    logging.info("Visualizing tokenization quality...")

    for idx in range(num_samples):
        original_text = dataset['output'][idx]
        tokenized_text = tokenizer(original_text)
        token_ids = tokenized_text['input_ids']
        detokenized_text = tokenizer.decode(token_ids, skip_special_tokens=True)
        
        logging.info(f"Sample {idx+1}:")
        logging.info(f"Original Text: {original_text}")
        logging.info(f"Tokenized (IDs): {token_ids}")
        logging.info(f"Detokenized Text: {detokenized_text}")
        logging.info("-" * 80)


def final_dataset_loader(dataset_path):
    raw_dataset = load_dataset(dataset_path)
    if raw_dataset is None:
        logging.error("Failed to load dataset.")
        exit(1)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token as EOS token

    tokenized_data = preprocess_and_tokenize(raw_dataset, tokenizer)

    visualize_tokenization_quality(raw_dataset, tokenizer) #visualizing tokeinization for first few samples

    alpaca_dataset = AlpacaDataset(tokenized_data)

    dataloader = create_dataloader(alpaca_dataset)

    return dataloader



if __name__ == "__main__":
    dataset_path = "/Users/Apple/Desktop/Pytorch/PyTorch_finetune_GPT2/dataset/alpaca_data.json"

    dataloader = final_dataset_loader(dataset_path)


    for batch in dataloader:
        logging.info(f"Batch - input_ids shape: {batch['input_ids'].shape}")
        break  # Only fetching the first batch for testing
