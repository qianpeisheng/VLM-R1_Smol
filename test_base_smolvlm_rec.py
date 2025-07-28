#!/usr/bin/env python3
"""
Quick test to verify the base SmolVLM-Instruct model can perform REC tasks
before starting the expensive LoRA training process.
"""

import sys
import os
import torch
from PIL import Image
import json

# Add VLM-R1 to path
sys.path.append('/nas/xulei/VLM-R1_Smol/src/open-r1-multimodal')

def test_base_smolvlm():
    print("Testing base SmolVLM-Instruct model on REC tasks...")
    
    # Import SmolVLM module
    try:
        from open_r1.vlm_modules.smolvlm_module import SmolVLMModule
        print("✓ SmolVLM module imported successfully")
    except Exception as e:
        print(f"✗ Failed to import SmolVLM module: {e}")
        return False
    
    # Load model using SmolVLM module
    model_path = "HuggingFaceTB/SmolVLM-Instruct"
    print(f"Loading model: {model_path}")
    
    try:
        module = SmolVLMModule()
        
        # Get model and processor classes
        model_cls = module.get_model_class(model_path, {})
        processor_cls = module.get_processing_class()
        
        # Load processor and model
        processor = processor_cls.from_pretrained(model_path)
        model = model_cls.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        print(f"Error details: {type(e).__name__}: {str(e)}")
        return False

    # Test with a sample REC task
    image_path = "/nas/xulei/VLM-R1_Smol/examples/image1.jpg"
    try:
        image = Image.open(image_path).convert('RGB')
        print(f"✓ Loaded test image: {image_path}")
    except Exception as e:
        print(f"✗ Failed to load image: {e}")
        return False

    # Test prompts in the format expected by the training
    test_prompts = [
        "Please provide the bounding box coordinate of the region this sentence describes: person on the left",
        "Please provide the bounding box coordinate of the region this sentence describes: red car",
        "Please provide the bounding box coordinate of the region this sentence describes: building in the background"
    ]

    print("\nTesting REC capabilities...")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest {i}: {prompt}")
        
        # Prepare conversation format
        conversation = [
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        try:
            # Process inputs
            text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(
                text=[text_prompt], 
                images=[image], 
                padding=True, 
                return_tensors="pt"
            ).to(model.device)
            
            # Generate response
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    repetition_penalty=1.2,
                    do_sample=True,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
            
            # Decode response
            generated_ids = output_ids[0][len(inputs.input_ids[0]):]
            response = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            print(f"Response: {response}")
            
            # Check if response looks like coordinates
            if any(char.isdigit() for char in response):
                print("✓ Response contains numbers (good sign)")
            else:
                print("⚠ Response doesn't contain numbers (may be issue)")
                
        except Exception as e:
            print(f"✗ Generation failed: {e}")
            return False
    
    print("\n" + "="*60)
    print("Base model test completed!")
    print("If the model produces reasonable responses (even if not perfect),")
    print("it indicates the base model has some REC capability that can be")
    print("improved with LoRA fine-tuning.")
    print("="*60)
    
    return True

if __name__ == "__main__":
    test_base_smolvlm()
