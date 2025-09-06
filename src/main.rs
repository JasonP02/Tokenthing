// Rust implementation of bpe

// Load dataset
use std::{fs, vec};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
struct Config {
    hf_dataset_names: String,
    tokenizer_vocab_size: u32,
    tokenizer_sequence_length: u32,
    tokenizer_save_path: String,
    file_path: String,
}

fn load_config(config_path: &str) -> Result<Config, Box<dyn std::error::Error>> {
    let config_content = fs::read_to_string(config_path)?;
    let config: Config = serde_yaml::from_str(&config_content)?;
    Ok(config)
}

fn load_and_validate_config() -> Config {
    let config_path = "/home/j/Projects/Tokenthing/cfg/config.yaml";
    match load_config(config_path) {
        Ok(config) => {
            println!("Config loaded successfully: {:?}", config);
            config
        }
        Err(e) => {
            eprintln!("Error loading config: {}", e);
            std::process::exit(1);
        }
    }
}

fn train_tokenizer(text: &str, vocab_size: u32, tokenizer_seq_len: u32) -> Vec<u8> {
    // Create initial vocab vector
    let _vocab_vector = vec![0; vocab_size as usize];

    // Process text in chunks of tokenizer_seq_len
    let mut current_chunk = String::new(); // New string type for chunking
    let mut chunk_count = 0; 
    
    for line in text.lines() {
        // Check if adding this line would exceed the sequence length
        if current_chunk.len() + line.len() + 1 > tokenizer_seq_len as usize {
            // If current chunk is not empty, process it
            if !current_chunk.is_empty() {
                chunk_count += 1;
                println!("Chunk {chunk_num} (length: {length}): {content}", 
                         chunk_num = chunk_count, 
                         length = current_chunk.len(), 
                         content = current_chunk);
                
                // TODO: Process this chunk for tokenizer training
                // process_chunk(&current_chunk);
                
                // Reset for next chunk
                current_chunk.clear(); 
            }
            
            // Handle the line that would exceed the sequence length
            // Calculate how much space we have left in current chunk
            let remaining_space = tokenizer_seq_len as usize - current_chunk.len();
            
            if remaining_space > 0 {
                // Add as much of the line as we can fit
                let (fits, overflow) = line.split_at(remaining_space);
                current_chunk.push_str(fits);
                
                // Process the current chunk (it's now full)
                chunk_count += 1;
                println!("Chunk {chunk_num} (length: {length}): {content}", 
                         chunk_num = chunk_count, 
                         length = current_chunk.len(), 
                         content = current_chunk);
                
                // Start new chunk with the overflow
                current_chunk.clear();
                current_chunk.push_str(overflow);
            } else {
                // Current chunk is already full, start new chunk with entire line
                current_chunk = line.to_string();
            }
        } else {
            // Add line to current chunk
            if !current_chunk.is_empty() {
                current_chunk.push('\n');
            }
            current_chunk.push_str(line);
        }
    }
    
    // Process the last chunk if it's not empty
    if !current_chunk.is_empty() {
        chunk_count += 1;
        println!("Final chunk {chunk_num} (length: {length}): {content}", 
                 chunk_num = chunk_count, 
                 length = current_chunk.len(), 
                 content = current_chunk);
        // TODO: Process this chunk for tokenizer training
        // process_chunk(&current_chunk);
    }
    
    println!("Total chunks processed: {}", chunk_count);
    
    // Return a copy of the input for now
    text.as_bytes().to_vec()
}

fn main() {
    // Load configuration
    let config = load_and_validate_config();

    // Loading text using config
    let content = fs::read_to_string(config.file_path);

    // match is 
    match content {
        Ok(text) => {
            train_tokenizer(&text, config.tokenizer_vocab_size, config.tokenizer_sequence_length);
        }
        Err(e) => {
            eprintln!("Error reading file: {}", e);
        }
    }

}

