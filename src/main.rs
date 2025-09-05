// Rust implementation of bpe

// Load dataset
use std::fs;
use std::u8;

fn process_text(text: &str) {
    println!("\n--- Lines of the file ---");
    for line in text.lines() {
        println!("- {}", line);
    }

    if text.contains("tokenizer") {
        println!("File contains 'tokenizer'")
    } else {
        println!("File does not contain 'tokenizer'")
    }
}

fn convert_text_to_utf_8(text: &str) -> &[u8] {
    let utf8_text: &[u8] = text.as_bytes();
    println!("{:?}", utf8_text);
    return utf8_text
}

fn main() {
    let file_path = "/home/j/Projects/Tokenthing/data/dataset.txt"; // Update this path to your actual dataset file
    let content = fs::read_to_string(file_path);

    match content {
        Ok(text) => {
            println!("File content:\n{}", text);
            process_text(&text);
            let _utf8_text = convert_text_to_utf_8(&text);
        }
        Err(e) => {
            eprintln!("Error reading file: {}", e);
        }
    }
}
