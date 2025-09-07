//! # Tokenizer Training Implementation
//! 
//! This module implements a Byte Pair Encoding (BPE) tokenizer training pipeline.
//! It loads text data, chunks it into manageable pieces, and processes it for tokenizer training.

use std::{fs, vec};
use serde::{Deserialize, Serialize};

/// Configuration structure for tokenizer training parameters.
/// 
/// This struct contains all the necessary configuration options for training a tokenizer,
/// including dataset information, vocabulary size, sequence length, and file paths.
#[derive(Debug, Deserialize, Serialize)]
struct Config {
    /// Name of the Hugging Face dataset to use for training
    hf_dataset_names: String,
    /// Target vocabulary size for the tokenizer
    tokenizer_vocab_size: u32,
    /// Maximum sequence length for tokenizer training
    tokenizer_sequence_length: u32,
    /// Path where the trained tokenizer will be saved
    tokenizer_save_path: String,
    /// Path to the local text file for training
    file_path: String,
}

/// Loads configuration from a YAML file.
fn load_config(config_path: &str) -> Result<Config, Box<dyn std::error::Error>> {
    let config_content = fs::read_to_string(config_path)?;
    let config: Config = serde_yaml::from_str(&config_content)?;
    Ok(config)
}

/// Loads and validates the configuration from the default config file.
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

/// This function takes raw text, chunks it into manageable pieces, and processes
/// those chunks for tokenizer training
fn train_tokenizer(text: &str, vocab_size: u32, tokenizer_seq_len: u32) -> Vec<u8> {
    let chunks = create_text_chunks(text, tokenizer_seq_len as usize);
    process_chunks(chunks);
    
    // Return a copy of the input for now
    text.as_bytes().to_vec()
}

/// Calculates how much space is available in the current chunk for adding content.
fn calculate_available_space(current_chunk: &str, max_chunk_size: usize) -> usize {
    let space_left = max_chunk_size.saturating_sub(current_chunk.len());
    let prefix_len = if current_chunk.is_empty() { 0 } else { 1 }; // newline
    space_left.saturating_sub(prefix_len)
}


/// Starts a new chunk, saving the current one if it has content.
fn start_new_chunk(chunks: &mut Vec<String>, current_chunk: String) -> String {
    if !current_chunk.is_empty() {
        chunks.push(current_chunk);
    }
    String::new()
}

/// Packs a line into chunks, splitting it across multiple chunks if necessary.
fn pack_chunk(line: &str, chunks: &mut Vec<String>, current_chunk: &mut String, max_chunk_size: usize) {
    let mut line_remaining = line;
    
    while !line_remaining.is_empty() {
        // If current chunk is full, start a new one
        if current_chunk.len() >= max_chunk_size {
            *current_chunk = start_new_chunk(chunks, std::mem::take(current_chunk));
        }
        
        // Calculate how much of the line we can fit
        let space_needed = if current_chunk.is_empty() { 0 } else { 1 }; // newline
        let space_available = max_chunk_size - current_chunk.len() - space_needed;
        
        if line_remaining.len() <= space_available {
            // Entire line fits - add it and we're done with this line
            if !current_chunk.is_empty() {
                current_chunk.push('\n');
            }
            current_chunk.push_str(line_remaining);
            break;
        } else {
            // Only part of line fits - add what we can and continue
            let (fits, rest) = line_remaining.split_at(space_available);
            if !current_chunk.is_empty() {
                current_chunk.push('\n');
            }
            current_chunk.push_str(fits);
            line_remaining = rest;
        }
    }
}

fn create_text_chunks(text: &str, max_chunk_size: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current_chunk = String::new();
    
    for line in text.lines() {
        pack_chunk(line, &mut chunks, &mut current_chunk, max_chunk_size);
    }
    
    if !current_chunk.is_empty() {
        chunks.push(current_chunk);
    }
    
    chunks
}

fn process_chunks(chunks: Vec<String>) {
    for (index, chunk) in chunks.iter().enumerate() {
        let chunk_num = index + 1;
        println!("Chunk {chunk_num} (length: {length}): {content}", 
                 chunk_num = chunk_num, 
                 length = chunk.len(), 
                 content = chunk);
        
        // TODO: Process this chunk for tokenizer training
        // process_chunk(chunk);
    }
    
    println!("Total chunks processed: {}", chunks.len());
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

