import yaml
import os
from datasets import load_dataset

def load_config():
    with open('cfg/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def download_and_save_datasets():
    config = load_config()
    dataset_names = config['hf_dataset_names']
    
    # Handle multiple datasets (comma-separated)
    if isinstance(dataset_names, str):
        dataset_names = [name.strip() for name in dataset_names.split(',')]
    
    for dataset_name in dataset_names:
        print(f"Loading dataset: {dataset_name}")
        
        try:
            # Load dataset
            dataset = load_dataset(dataset_name)
            
            # Get train split (default to first split if no 'train' exists)
            if 'train' in dataset:
                train_data = dataset['train']
            else:
                train_data = list(dataset.values())[0]
            
            # Extract text content
            texts = []
            for item in train_data:
                if 'text' in item:
                    texts.append(item['text'])
                elif 'content' in item:
                    texts.append(item['content'])
                else:
                    # Use the first string field if no 'text' or 'content'
                    for key, value in item.items():
                        if isinstance(value, str):
                            texts.append(value)
                            break
            
            # Create output filename
            safe_name = dataset_name.replace('/', '_')
            output_file = f"{safe_name}.txt"
            
            # Save to text file
            with open(output_file, 'w', encoding='utf-8') as f:
                for text in texts:
                    f.write(text + '\n')
            
            print(f"Saved {len(texts)} text samples to {output_file}")
            
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")

if __name__ == "__main__":
    download_and_save_datasets()