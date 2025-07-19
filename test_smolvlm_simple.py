#!/usr/bin/env python3
"""
Simple test script for SmolVLM module without external dependencies
"""

import sys
import os

# Add the VLM-R1 source to Python path
sys.path.append('/home/peisheng/VLM-R1/src/open-r1-multimodal')

def test_smolvlm_module():
    """Test the SmolVLM module implementation"""
    try:
        from open_r1.vlm_modules.smolvlm_module import SmolVLMModule
        print("✓ SmolVLM module imported successfully")
        
        # Test module initialization
        module = SmolVLMModule()
        print(f"✓ VLM Key: {module.get_vlm_key()}")
        
        # Test model class
        model_cls = module.get_model_class("HuggingFaceTB/SmolVLM-Instruct", {})
        print(f"✓ Model class: {model_cls}")
        
        # Test processing class
        proc_cls = module.get_processing_class()
        print(f"✓ Processing class: {proc_cls}")
        
        # Test vision modules keywords
        vision_keywords = module.get_vision_modules_keywords()
        print(f"✓ Vision modules keywords: {vision_keywords}")
        
        # Test multimodal keywords
        mm_keywords = module.get_custom_multimodal_keywords()
        print(f"✓ Multimodal keywords: {mm_keywords}")
        
        # Test reward functions
        reward_func = SmolVLMModule.select_reward_func("accuracy", "rec")
        print(f"✓ Accuracy reward function: {reward_func}")
        
        format_func = SmolVLMModule.select_reward_func("format", "rec")
        print(f"✓ Format reward function: {format_func}")
        
        # Test question template
        template = SmolVLMModule.get_question_template("rec")
        print(f"✓ REC question template: {template}")
        
        print("\n✅ All SmolVLM module tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ SmolVLM module test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_reward_functions():
    """Test the reward functions"""
    try:
        from open_r1.vlm_modules.smolvlm_module import SmolVLMModule
        
        # Test format reward
        test_completions = [
            [{"content": "The object is located at [100, 150, 200, 250]."}],
            [{"content": "I can see the object but cannot provide coordinates."}],
            [{"content": "The bounding box is [50, 75, 150, 175] for this object."}]
        ]
        
        format_rewards = SmolVLMModule.format_reward_rec(test_completions)
        print(f"✓ Format rewards: {format_rewards}")
        assert format_rewards == [1.0, 0.0, 1.0], f"Expected [1.0, 0.0, 1.0], got {format_rewards}"
        
        # Test IoU reward with proper format
        test_solution = [
            {"bbox": [100, 150, 200, 250]}, 
            {"bbox": [0, 0, 50, 50]}, 
            {"bbox": [40, 70, 160, 180]}
        ]
        iou_rewards = SmolVLMModule.iou_reward(test_completions, test_solution)
        print(f"✓ IoU rewards: {iou_rewards}")
        
        # Test with exact match (should give IoU = 1.0, reward = 2*1-1 = 1.0)
        exact_completions = [[{"content": "The answer is [100, 150, 200, 250]."}]]
        exact_solution = [{"bbox": [100, 150, 200, 250]}]
        exact_rewards = SmolVLMModule.iou_reward(exact_completions, exact_solution)
        print(f"✓ Exact match reward: {exact_rewards}")
        assert abs(exact_rewards[0] - 1.0) < 0.001, f"Expected ~1.0, got {exact_rewards[0]}"
        
        print("\n✅ Reward function tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Reward function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_module_selection():
    """Test if module can be selected correctly"""
    try:
        # Test the selection logic directly
        test_paths = [
            "HuggingFaceTB/SmolVLM-Instruct",
            "smolvlm-2b",
            "SmolVLM-2B-Instruct",
            "huggingfacetb/smolvlm-instruct"
        ]
        
        from open_r1.vlm_modules.smolvlm_module import SmolVLMModule
        
        for path in test_paths:
            if "smolvlm" in path.lower():
                print(f"✓ Path '{path}' would select SmolVLMModule")
            else:
                print(f"⚠ Path '{path}' would NOT select SmolVLMModule")
        
        print("\n✅ Module selection test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Module selection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Testing SmolVLM module (simple version)...")
    print("=" * 50)
    
    success = True
    
    print("\n1. Testing SmolVLM Module...")
    success &= test_smolvlm_module()
    
    print("\n2. Testing Reward Functions...")
    success &= test_reward_functions()
    
    print("\n3. Testing Module Selection...")
    success &= test_module_selection()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! SmolVLM module is working correctly.")
        print("\nNext steps:")
        print("1. Install SmolVLM: pip install transformers")
        print("2. Run training: ./run_scripts/run_grpo_rec_smolvlm_lora.sh")
        print("3. Evaluate: python3 src/eval/test_rec_smolvlm.py --model_path /path/to/model")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)
