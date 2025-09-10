use std::{collections::HashMap, env::current_exe, fs, io::{BufRead, BufReader}};
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
    current: &mut String,
    max: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // Centralized: always flush the same way
    fn flush(chunks: &mut Vec<String>, cur: &mut String) {
        if !cur.is_empty() {
            chunks.push(std::mem::take(cur));
        }
    }

    // Centralized: append piece, adding a newline iff chunk has prior content
    fn append_piece(cur: &mut String, piece: &str) {
        if !cur.is_empty() {
            cur.push('\n');
        }
        cur.push_str(piece);
    }

    let mut remaining = line;

    while !remaining.is_empty() {
        if current.len() >= max {
            flush(chunks, current);
            continue;
        }

        let needs_nl = !current.is_empty();
        let nl_cost = if needs_nl { 1 } else { 0 };

        // If no space left even for the newline, flush and retry
        if current.len() + nl_cost >= max {
            flush(chunks, current);
            continue;
        }

        let space = max - current.len() - nl_cost;

        // Fast path: whole remainder fits
        if remaining.len() <= space {
            append_piece(current, remaining);
            break;
        }

        // Take largest valid UTF-8 prefix within byte budget
        let mut end = space.min(remaining.len());
        while end > 0 && !remaining.is_char_boundary(end) {
            end -= 1;
        }

        if end == 0 {
            // Couldn’t fit even one char (after accounting for newline) → flush
            flush(chunks, current);
            continue;
        }

        append_piece(current, &remaining[..end]);
        remaining = &remaining[end..];
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