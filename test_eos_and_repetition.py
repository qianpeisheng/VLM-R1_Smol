#!/usr/bin/env python3
"""
Quick test to diagnose EOS token and repetitive generation issues
"""

import sys
import os
import json
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, StoppingCriteria, StoppingCriteriaList
from PIL import Image

# Add VLM-R1 source to path
sys.path.append('/nas/xulei/VLM-R1_Smol/src/open-r1-multimodal')

from open_r1.vlm_modules.smolvlm_module import SmolVLMModule

class RepetitionStoppingCriteria(StoppingCriteria):
    """Stop generation when repetitive patterns are detected"""
    def __init__(self, tokenizer, max_repetitions=3):
        self.tokenizer = tokenizer
        self.max_repetitions = max_repetitions
        
    def __call__(self, input_ids, scores, **kwargs):
        # Check if last 60 tokens contain repetitive patterns
        if len(input_ids[0]) > 60:
            last_text = self.tokenizer.decode(input_ids[0][-60:], skip_special_tokens=False)
            
            # Check for repetitive "Answer:" pattern
            answer_count = last_text.count("Answer:")
            if answer_count > self.max_repetitions:
                print(f"🛑 Stopping due to repetitive 'Answer:' pattern ({answer_count} times)")
                return True
                
            # Check for repetitive JSON structures
            brace_count = last_text.count('{\n  "answer":')
            if brace_count > self.max_repetitions:
                print(f"🛑 Stopping due to repetitive JSON pattern ({brace_count} times)")
                return True
                
        return False

def check_tokenizer_config(model_path):
    """Check SmolVLM tokenizer configuration"""
    print("🔍 Checking SmolVLM Tokenizer Configuration")
    print("=" * 50)
    
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    
    print(f"Model: {model_path}")
    print(f"EOS token: '{tokenizer.eos_token}'")
    print(f"EOS token ID: {tokenizer.eos_token_id}")
    print(f"PAD token: '{tokenizer.pad_token}'")
    print(f"PAD token ID: {tokenizer.pad_token_id}")
    print(f"BOS token: '{tokenizer.bos_token}'")
    print(f"BOS token ID: {tokenizer.bos_token_id}")
    
    # Test encoding/decoding
    test_text = "This is a test <answer>[100, 200, 300, 400]</answer>"
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    
    print(f"\nTest encoding:")
    print(f"Original: {test_text}")
    print(f"Tokens: {tokens[-10:]} (last 10)")
    print(f"Decoded: {decoded}")
    print(f"Last token is EOS: {tokens[-1] == tokenizer.eos_token_id}")
    
    return processor

def test_generation_with_stopping(model_path, processor):
    """Test generation with repetition stopping criteria"""
    print("\n🧪 Testing Generation with Stopping Criteria")
    print("=" * 50)
    
    # Load model
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    
    # Load SmolVLM module for proper input preparation
    vlm_module = SmolVLMModule()
    vlm_module.post_model_init(model, processor)
    
    # Test with a simple image and prompt
    test_prompt = "Please provide the bounding box coordinate of the region this sentence describes: person in the image. First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
    
    # Create a dummy image (you can replace with a real one)
    dummy_image = Image.new('RGB', (640, 480), color='white')
    
    print(f"Test prompt: {test_prompt[:100]}...")
    
    # Test different generation configurations
    configs = [
        {
            "name": "Original (likely to repeat)",
            "kwargs": {
                "max_new_tokens": 256,
                "do_sample": False,
                "temperature": 0.0,
                "use_cache": True
            }
        },
        {
            "name": "With EOS and repetition penalty",
            "kwargs": {
                "max_new_tokens": 256,
                "do_sample": False,
                "temperature": 0.0,
                "eos_token_id": processor.tokenizer.eos_token_id,
                "pad_token_id": processor.tokenizer.eos_token_id,
                "repetition_penalty": 1.2,
                "use_cache": True
            }
        },
        {
            "name": "With stopping criteria",
            "kwargs": {
                "max_new_tokens": 256,
                "do_sample": False,
                "temperature": 0.0,
                "eos_token_id": processor.tokenizer.eos_token_id,
                "pad_token_id": processor.tokenizer.eos_token_id,
                "repetition_penalty": 1.1,
                "stopping_criteria": StoppingCriteriaList([RepetitionStoppingCriteria(processor.tokenizer)]),
                "use_cache": True
            }
        }
    ]
    
    for config in configs:
        print(f"\n📋 Testing: {config['name']}")
        print("-" * 40)
        
        try:
            # Prepare inputs
            model_inputs, _ = vlm_module.prepare_model_inputs(
                processor, 
                [f"<image> {test_prompt}"], 
                [dummy_image],
                return_tensors="pt",
                padding=True,
                padding_side="left"
            )
            
            # Move to device
            model_inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                          for k, v in model_inputs.items()}
            
            # Generate
            with torch.no_grad():
                generated_ids = model.generate(**model_inputs, **config["kwargs"])
            
            # Decode
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in 
                zip(model_inputs["input_ids"], generated_ids)
            ]
            
            response = processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            # Analyze response
            answer_count = response.count("Answer:")
            json_count = response.count('{"answer":')
            
            print(f"Response length: {len(response)} chars")
            print(f"'Answer:' repetitions: {answer_count}")
            print(f"JSON structure repetitions: {json_count}")
            
            # Show first part of response
            print(f"First 300 chars:")
            print(response[:300])
            
            # Check if it looks repetitive
            if answer_count > 3 or json_count > 3:
                print("❌ REPETITIVE GENERATION DETECTED")
            else:
                print("✅ Generation looks normal")
                
        except Exception as e:
            print(f"❌ Error: {e}")

def check_training_data_eos():
    """Check if training data has EOS tokens"""
    print("\n🔍 Checking Training Data for EOS Tokens")
    print("=" * 50)
    
    # Check if there's a JSONL file (typical for VLM-R1 training)
    jsonl_files = [
        "/nas/xulei/VLM-R1_Smol/rec_jsons_processed/refcoco_train.jsonl",
        "/nas/xulei/VLM-R1_Smol/runs/smolvlm-instruct-rec-grpo-lora-optimized/train_data.jsonl"
    ]
    
    # Also check the original JSON files
    json_files = [
        "/nas/xulei/VLM-R1_Smol/rec_jsons_processed/refcoco_train.json"
    ]
    
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct", trust_remote_code=True)
    eos_token = processor.tokenizer.eos_token
    
    for file_path in jsonl_files + json_files:
        if os.path.exists(file_path):
            print(f"\n📁 Checking: {file_path}")
            print("-" * 30)
            
            try:
                if file_path.endswith('.jsonl'):
                    # JSONL format
                    with open(file_path, 'r') as f:
                        for i, line in enumerate(f):
                            if i >= 3:  # Check first 3 examples
                                break
                            data = json.loads(line)
                            
                            # Check completion field
                            if "completion" in data:
                                completion = data["completion"]
                                has_eos = completion.endswith(eos_token)
                                print(f"Example {i}: EOS token present: {has_eos}")
                                print(f"  Last 50 chars: ...{completion[-50:]}")
                                
                elif file_path.endswith('.json'):
                    # JSON format
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        
                    for i in range(min(3, len(data))):
                        sample = data[i]
                        print(f"Example {i}:")
                        print(f"  Keys: {list(sample.keys())}")
                        
                        # Check if there's solution text that might need EOS
                        if 'solution' in sample:
                            solution = sample['solution']
                            if isinstance(solution, str):
                                has_eos = solution.endswith(eos_token)
                                print(f"  Solution has EOS: {has_eos}")
                                print(f"  Solution: {solution}")
                        
            except Exception as e:
                print(f"❌ Error reading {file_path}: {e}")
        else:
            print(f"❌ File not found: {file_path}")

def main():
    """Main test function"""
    model_path = "/nas/xulei/VLM-R1_Smol/checkpoints/rl/smolvlm-instruct-rec-grpo-lora-optimized/checkpoint-7400"
    
    print("🚀 SmolVLM EOS Token and Repetition Debug Test")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please update the model_path in the script")
        return
    
    # Step 1: Check tokenizer configuration
    processor = check_tokenizer_config(model_path)
    
    # Step 2: Test generation with different stopping criteria
    test_generation_with_stopping(model_path, processor)
    
    # Step 3: Check training data for EOS tokens
    check_training_data_eos()
    
    print("\n✅ Debug test completed!")
    print("\n📋 Summary:")
    print("1. Check if EOS token configuration looks correct")
    print("2. See which generation config reduces repetition")
    print("3. Verify if training data needs EOS tokens added")

if __name__ == "__main__":
    main()
