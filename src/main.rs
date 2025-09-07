//! # Tokenizer Training Implementation
//! 
//! This module implements a Byte Pair Encoding (BPE) tokenizer training pipeline.
//! It loads text data, chunks it into manageable pieces, and processes it for tokenizer training.

use std::{fs, vec, io::{BufRead, BufReader}};
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

/// This function trains a tokenizer on text data using a streaming approach
fn train_tokenizer(file_path: &str, vocab_size: u32, tokenizer_seq_len: u32) -> Result<(), Box<dyn std::error::Error>> {
    train_tokenizer_streaming(file_path, vocab_size, tokenizer_seq_len)
}

fn train_tokenizer_streaming(file_path: &str, vocab_size: u32, tokenizer_seq_len: u32) -> Result<(), Box<dyn std::error::Error>> {
    let file = fs::File::open(file_path)?;
    let reader = BufReader::new(file);
    
    let mut current_chunk = String::new();
    let mut completed_chunks: Vec<String> = Vec::new();
    const BATCH_SIZE: usize = 100;
    
    for line_result in reader.lines() {
        let line = line_result?;
        
        // Reuse the existing chunking logic to fill completed_chunks
        pack_chunk(&line, &mut completed_chunks, &mut current_chunk, tokenizer_seq_len as usize);
        
        // When we have enough completed chunks, process them and clear the buffer
        if completed_chunks.len() >= BATCH_SIZE {
            process_chunks(std::mem::take(&mut completed_chunks));
        }
    }
    
    // Flush the final in-progress chunk if it has content
    if !current_chunk.is_empty() {
        completed_chunks.push(current_chunk);
    }
    
    // Process any remaining completed chunks
    if !completed_chunks.is_empty() {
        process_chunks(completed_chunks);
    }
    
    Ok(())
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

fn process_chunks(chunks: Vec<String>) {
    for (index, chunk) in chunks.iter().enumerate() {
        let chunk_num = index + 1;
        println!("Chunk {chunk_num} (length: {length}): {content}", 
                 chunk_num = chunk_num, 
                 length = chunk.len(), 
                 content = chunk);
        
        for chunk in chunks {
            // iterate over text chunks and break them down into words with regex... or - should we split all words at once with regex
            // now that i think about it, why do we have multiple 'chunks' if we are going to be processing as much text as possible with a regex pass?
        }
    }
    
    println!("Total chunks processed: {}", chunks.len());
}

fn main() {
    // Load configuration
    let config = load_and_validate_config();

    // Train tokenizer (automatically chooses between in-memory and streaming approaches)
    match train_tokenizer(&config.file_path, config.tokenizer_vocab_size, config.tokenizer_sequence_length) {
        Ok(_) => {
            println!("Tokenizer training completed successfully");
        }
        Err(e) => {
            eprintln!("Error during tokenizer training: {}", e);
            std::process::exit(1);
        }
    }
}

