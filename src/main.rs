use std::{collections::{HashMap}, fs, io::{BufRead, BufReader}};
use serde::{Deserialize, Serialize};
use regex::Regex;

type TokenPair = (String,String);
type PairFreqs = HashMap<TokenPair, u32>;
type ResultE = Result<(), Box<dyn std::error::Error>>;

#[derive(Debug, Deserialize, Serialize)]
struct Config {
    hf_dataset_names: String,
    file_path: String,
    tokenizer_vocab_size: usize,
    tokenizer_sequence_length: usize,
}

fn load_config() -> Result<Config, Box<dyn std::error::Error>> {
    let config_path = "/home/j/Projects/Tokenthing/cfg/config.yaml";
    let config_content = fs::read_to_string(config_path)?;
    let config = serde_yaml::from_str(&config_content)?;
    Ok(config)
}

// Apply learned merges to a token sequence.
// Performs greedy left-to-right passes using the learned pair ranks.
fn apply_merges_to_tokens(mut tokens: Vec<String>, merges: &[(String, String)]) -> Vec<String> {
    if merges.is_empty() || tokens.len() < 2 {
        return tokens;
    }

    let mut ranks: HashMap<(String, String), usize> = HashMap::new();
    for (i, (a, b)) in merges.iter().enumerate() {
        ranks.insert((a.clone(), b.clone()), i);
    }

    loop {
        if tokens.len() < 2 { break; }
        let mut i = 0;
        let mut merged_any = false;
        while i + 1 < tokens.len() {
            let pair = (tokens[i].clone(), tokens[i + 1].clone());
            if ranks.contains_key(&pair) {
                let new_tok = format!("{}{}", pair.0, pair.1);
                tokens[i] = new_tok;
                tokens.remove(i + 1);
                merged_any = true;
            } else {
                i += 1;
            }
        }
        if !merged_any { break; }
    }

    tokens
}

fn count_token_pairs(tokens: &[String]) -> PairFreqs {
    let mut pair_freqs = PairFreqs::new();
    
    for window in tokens.windows(2) {
        let pair = (window[0].clone(), window[1].clone());
        *pair_freqs.entry(pair).or_insert(0) += 1;
    }
    pair_freqs
}

// Map step: tokenize a text slice, apply current merges, then count pairs
fn map_count_pairs(text: &str, re: &Regex, merges: &[(String, String)]) -> PairFreqs {
    let base_tokens: Vec<String> = pretokenize(re, text).map(str::to_string).collect();
    let tokens = apply_merges_to_tokens(base_tokens, merges);
    count_token_pairs(&tokens)
}


fn apply_regex() -> Regex {
    Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]+|[^\s\p{L}\p{N}]+|\s+").unwrap()
}

fn pretokenize<'a>(pat: &'a Regex, text: &'a str) -> impl Iterator<Item = &'a str> + 'a {
    pat.find_iter(text).map(|m| m.as_str())
}
fn train_tokenizer(
    file_path: &str,
    vocab_size: usize,
    _seq_len: usize) -> ResultE {
    // Learned merges and a simple score for merged tokens when discovered
    let mut merges: Vec<TokenPair> = Vec::new();
    let mut vocab: HashMap<String, u32> = HashMap::new();

    let re = apply_regex();

    // Repeat passes until reaching vocab_size or no pairs remain
    loop {
        if merges.len() >= vocab_size { break; }

        let file = fs::File::open(file_path)?;
        let mut reader = BufReader::new(file);
        let mut line = String::new();
        let mut global_counts: PairFreqs = PairFreqs::new();

        loop {
            line.clear();
            let n = reader.read_line(&mut line)?;
            if n == 0 { break; }
            if line.ends_with('\n') { line.pop(); if line.ends_with('\r') { line.pop(); } }

            let counts = map_count_pairs(&line, &re, &merges);
            for (k, v) in counts { *global_counts.entry(k).or_insert(0) += v; }
        }

        let best = global_counts
            .iter()
            .max_by_key(|(_, &freq)| freq)
            .map(|(pair, &freq)| (pair.clone(), freq));

        match best {
            Some((pair, freq)) if freq > 0 => {
                let merged = format!("{}{}", pair.0, pair.1);
                vocab.insert(merged, freq);
                merges.push(pair);
            }
            _ => { println!("No more pairs to merge."); break; }
        }
    }

    println!("Learned {} merges", merges.len());
    Ok(())
}


fn main() -> ResultE {
    let config = load_config()?;
    train_tokenizer(
        &config.file_path,
        config.tokenizer_vocab_size,
        config.tokenizer_sequence_length,
    )?;
    
    Ok(())
}
