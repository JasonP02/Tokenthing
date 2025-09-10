use std::{fs, io::{BufRead, BufReader}, collections::HashMap};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
struct Config {
    hf_dataset_names: String,
    file_path: String,
    tokenizer_vocab_size: u32,
    tokenizer_sequence_length: usize,
}

fn load_config() -> Result<Config, Box<dyn std::error::Error>> {
    let config_path = "/home/j/Projects/Tokenthing/cfg/config.yaml";
    let config_content = fs::read_to_string(config_path)?;
    let config = serde_yaml::from_str(&config_content)?;
    Ok(config)
}

fn pack_chunk(
    line: &str,
    chunks: &mut Vec<String>,
    current_chunk: &mut String,
    max_chunk_size: usize,)
    -> Result<(), Box<dyn std::error::Error>> {
    // Function for packing a mutable 'chunk' from lines of a text file into a max chunk size
    let mut line_remaining = line;

    while !line_remaining.is_empty() {
        // if chunk full, make a new chunk
        if current_chunk.len() >= max_chunk_size {
            chunks.push(current_chunk.drain(..).collect());
        }

        let line_prefix = if current_chunk.is_empty() {0} else {1}; // adding \n for additional lines in the same chunk
        let space_available = max_chunk_size - current_chunk.len() - line_prefix; // calculation for how much text we can add

        current_chunk.push_str(line_remaining);

        if line_remaining.chars().count() <= space_available {
            current_chunk.push_str(line_remaining);
            break;
        } else {
            // if we will have overflow, we need to split by char
            let fits = &line_remaining[..space_available.min(line_remaining.len())];
            let rest = &line_remaining[fits.len()..];

            current_chunk.push_str(fits);
            line_remaining = rest;
            }
        }
    Ok(())
}

fn process_chunks(
    chunks: Vec<String>,
    vocab_size: u32,
) -> Result<(), Box<dyn std::error::Error>> {

}

fn train_tokenizer(
    file_path: &str,
    vocab_size: u32,
    seq_len: usize,
) -> Result<(), Box<dyn std::error::Error>> {
        let file = fs::File::open(file_path)?;
        let reader = BufReader::new(file);

        let mut current_chunk = String::new();
        let mut completed_chunks : Vec<String> = Vec::new();
        let mut vocab: HashMap<String, u32> = HashMap::new();

        const BATCH_SIZE: usize = 100;

        for line_result in reader.lines() {
            let line = line_result?;
            pack_chunk(&line, &mut completed_chunks, &mut current_chunk, seq_len)?;

            if completed_chunks.len() > BATCH_SIZE {
                // take the completed chunks from memory for us to use and to allow pack_chunk to keep packing

                process_chunks(std::mem::take(&mut completed_chunks), vocab_size);
            }
        }
        Ok(())
    }

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = load_config()?;
    train_tokenizer(
        &config.file_path,
        config.tokenizer_vocab_size,
        config.tokenizer_sequence_length,
    )?;
    
    Ok(())
}