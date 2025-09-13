# BPE Tokenizer Implementation Guide

## Understanding BPE Training (The Core Concept)

**BPE (Byte Pair Encoding)** is surprisingly simple once you grasp the core idea:

1. **Start Small**: Begin with basic tokens (characters or pretokenized words)
2. **Find Patterns**: Look for the most frequent adjacent token pairs
3. **Create New Tokens**: Merge frequent pairs into single new tokens
4. **Repeat**: Keep merging until you reach your vocabulary size limit

The key insight: **You're not "learning" new tokens - you're discovering optimal merges of existing ones.**

## Your Current Code Analysis

### What's Working Well:
- ✅ Text chunking with [`pack_chunk()`](src/main.rs:20)
- ✅ Pretokenization with regex
- ✅ Basic vocabulary frequency counting
- ✅ Pair frequency counting

### What's Missing:
- ❌ **The actual BPE merge loop** - this is the heart of the algorithm
- ❌ Vocabulary growth mechanism
- ❌ Token-to-ID mapping
- ❌ Model serialization

## The Missing BPE Algorithm (Step-by-Step)

### Step 1: Find Most Frequent Pair
```rust
fn find_most_frequent_pair(pair_freqs) -> Option<(String, String)> {
    pair_freqs.iter()
        .max_by_key(|(_, &freq)| freq)
        .map(|(pair, _)| pair.clone())
}
```

### Step 2: Merge Tokens
```rust
fn merge_tokens(tokens: Vec<String>, pair: &(String, String), new_token: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut i = 0;
    
    while i < tokens.len() {
        if i < tokens.len() - 1 && 
           tokens[i] == pair.0 && tokens[i + 1] == pair.1 {
            result.push(new_token.to_string());
            i += 2;
        } else {
            result.push(tokens[i].clone());
            i += 1;
        }
    }
    result
}
```

### Step 3: The Complete BPE Loop
```rust
fn perform_bpe_iterations(
    mut all_tokens: Vec<Vec<String>>,
    target_vocab_size: usize,
    initial_vocab: HashSet<String>
) -> (HashMap<String, u32>, Vec<Vec<String>>) {
    let mut vocab = initial_vocab;
    let mut merge_rules = Vec::new();
    
    while vocab.len() < target_vocab_size {
        // 1. Count all adjacent pairs across all documents
        let mut pair_freqs = HashMap::new();
        for tokens in &all_tokens {
            for window in tokens.windows(2) {
                let pair = (window[0].clone(), window[1].clone());
                *pair_freqs.entry(pair).or_insert(0) += 1;
            }
        }
        
        // 2. Find most frequent pair
        let best_pair = match find_most_frequent_pair(&pair_freqs) {
            Some(pair) => pair,
            None => break, // No more pairs found
        };
        
        // 3. Create new token
        let new_token = format!("{}{}", best_pair.0, best_pair.1);
        
        // 4. Add to vocabulary
        vocab.insert(new_token.clone());
        
        // 5. Store merge rule
        merge_rules.push((best_pair.clone(), new_token.clone()));
        
        // 6. Apply merge to all documents
        for tokens in &mut all_tokens {
            *tokens = merge_tokens(tokens.clone(), &best_pair, &new_token);
        }
        
        println!("Vocab size: {}, New token: {}", vocab.len(), new_token);
    }
    
    // Create token-to-ID mapping
    let mut vocab_map = HashMap::new();
    for (id, token) in vocab.iter().enumerate() {
        vocab_map.insert(token.clone(), id as u32);
    }
    
    (vocab_map, all_tokens)
}
```

## Fixing Your process_chunks Function

Replace your current [`process_chunks()`](src/main.rs:85) with this complete implementation:

```rust
fn process_chunks(
    chunks: Vec<String>,
    vocab_size: usize,
) -> Result<HashMap<String, u32>, Box<dyn std::error::Error>> {
    let re = apply_regex();
    let mut all_tokens = Vec::new();
    
    // Step 1: Initial pretokenization
    for chunk in chunks {
        let tokens: Vec<String> = pretokenize(&re, &chunk)
            .map(|s| s.to_string())
            .collect();
        all_tokens.push(tokens);
    }
    
    // Step 2: Build initial vocabulary (unique tokens)
    let mut initial_vocab = HashSet::new();
    for tokens in &all_tokens {
        for token in tokens {
            initial_vocab.insert(token.clone());
        }
    }
    
    println!("Initial vocabulary size: {}", initial_vocab.len());
    
    // Step 3: Perform BPE iterations
    let (vocab_map, final_tokens) = perform_bpe_iterations(
        all_tokens,
        vocab_size,
        initial_vocab
    );
    
    println!("Final vocabulary size: {}", vocab_map.len());
    
    Ok(vocab_map)
}
```

## Encoding Function (Fixed)

Your [`encode_bpe()`](src/main.rs:121) function needs to apply the learned merge rules:

```rust
fn encode_bpe(
    text: &str,
    merge_rules: &[(String, String)],  // Learned merge rules
    vocab: &HashMap<String, u32>,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let re = apply_regex();
    let mut tokens: Vec<String> = pretokenize(&re, text)
        .map(|s| s.to_string())
        .collect();
    
    // Apply merge rules in order
    for (pair, merged_token) in merge_rules {
        tokens = merge_tokens(tokens, &(pair.0.clone(), pair.1.clone()), merged_token);
    }
    
    // Convert tokens to IDs
    let token_ids: Vec<u32> = tokens
        .iter()
        .filter_map(|token| vocab.get(token).copied())
        .collect();
    
    Ok(token_ids)
}
```

## Next Steps for You

1. **Start Small**: Test with just 10-20 lines of text first
2. **Debug Print**: Add lots of `println!` statements to see what's happening
3. **Visualize**: Print out the merge rules so you can see the patterns being learned
4. **Iterate**: Don't try to get 30k vocabulary on first try - start with 100

## Simple Test Example

Try this tiny dataset first:
```
"the cat sat on the mat"
"the dog sat on the log"
"a cat and a dog"
```

Expected BPE merges:
1. "th" + "e" → "the" (most frequent pair)
2. "sa" + "t" → "sat" 
3. "o" + "n" → "on"

This will help you see the algorithm working before scaling up.

## Common Rust Gotchas for BPE

1. **String vs &str**: You'll need lots of `.to_string()` calls
2. **HashMap cloning**: Be careful with performance when vocab gets large
3. **UTF-8 boundaries**: Your chunking code already handles this well
4. **Memory usage**: Consider streaming for large datasets

Remember: **BPE is iterative and greedy** - it makes the best merge at each step, not necessarily the globally optimal sequence. This is why it's fast and works well in practice!

Start with the small test case, get that working, then scale up. You've got the hard parts (chunking, pretokenization) already done!