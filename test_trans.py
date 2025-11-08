"""
IndicTrans2 Local Inference Script
Based on the official HuggingFace interface from AI4Bharat
Works with indictrans2-indic-en-dist-200M model
"""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor  # Corrected import
import os

# Configuration
MODEL_PATH = r"D:\real_time_live_translation\models\indictrans2-indic-en-dist-200M"
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def initialize_model_and_tokenizer(model_path, device):
    """Initialize model and tokenizer from local path"""
    print(f"Loading model from: {model_path}")
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True
    )
    
    # Load model
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        local_files_only=True
    )
    
    # Move model to device and set precision
    model = model.to(device)
    if device == "cuda":
        model.half()  # Use half precision on GPU
    
    model.eval()
    
    print(f"✓ Model loaded successfully")
    print(f"  Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    
    return tokenizer, model

def batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip):
    """Translate a batch of sentences"""
    translations = []
    
    for i in range(0, len(input_sentences), BATCH_SIZE):
        batch = input_sentences[i : i + BATCH_SIZE]
        
        # Preprocess the batch using IndicProcessor
        batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)
        
        # Tokenize the batch
        inputs = tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(DEVICE)
        
        # Generate translations
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                # --- THIS IS THE FIX ---
                use_cache=False,  # Set to False to bypass the CPU caching bug
                # ---------------------
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )
        
        # Decode the generated tokens
        generated_tokens = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        
        # Postprocess the translations
        translations += ip.postprocess_batch(generated_tokens, lang=tgt_lang)
        
        # Clean up memory
        del inputs
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    
    return translations

def main():
    print("="*60)
    print("IndicTrans2 Local Inference")
    print("="*60)
    
    # Check if model directory exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model directory not found at {MODEL_PATH}")
        print("Please ensure the model is downloaded to the correct location.")
        return
    
    # Initialize model and tokenizer
    try:
        tokenizer, model = initialize_model_and_tokenizer(MODEL_PATH, DEVICE)
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure all model files are present")
        print("2. Check if you have enough memory")
        print("3. Verify IndicTransToolkit is installed")
        return
    
    # Initialize IndicProcessor
    try:
        ip = IndicProcessor(inference=True)
        print("✓ IndicProcessor initialized")
    except Exception as e:
        print(f"Failed to initialize IndicProcessor: {e}")
        print("Please ensure IndicTransToolkit is properly installed:")
        print("pip install git+https://github.com/VarunGumma/IndicTransToolkit.git")
        return
    
    # Language configuration
    src_lang = "hin_Deva"  # Hindi in Devanagari script
    tgt_lang = "eng_Latn"  # English in Latin script
    
    # Test sentences
    test_sentences = [
        "जब मैं छोटा था, मैं हर रोज़ पार्क जाता था।",
        "हमने पिछले सप्ताह एक नई फिल्म देखी जो कि बहुत प्रेरणादायक थी।",
        "अगर तुम मुझे उस समय पास मिलते, तो हम बाहर खाना खाने चलते।",
        "मेरे मित्र ने मुझे उसके जन्मदिन की पार्टी में बुलाया है, और मैं उसे एक तोहफा दूंगा।",
    ]
    
    print("\n" + "="*60)
    print(f"Translation Examples ({src_lang} → {tgt_lang})")
    print("="*60)
    
    # Translate test sentences
    try:
        translations = batch_translate(
            test_sentences,
            src_lang,
            tgt_lang,
            model,
            tokenizer,
            ip
        )
        
        print()
        for input_sentence, translation in zip(test_sentences, translations):
            print(f"Hindi: {input_sentence}")
            print(f"English: {translation}")
            print("-" * 40)
    
    except Exception as e:
        print(f"Error during translation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Interactive mode
    print("\n" + "="*60)
    print("Interactive Translation Mode")
    print("Type Hindi sentences to translate (or 'quit' to exit)")
    print("="*60 + "\n")
    
    while True:
        try:
            user_input = input("Hindi text: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Translate single sentence
            translations = batch_translate(
                [user_input],
                src_lang,
                tgt_lang,
                model,
                tokenizer,
                ip
            )
            
            if translations:
                print(f"English: {translations[0]}\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")
    
    # Clean up
    del model, tokenizer
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()