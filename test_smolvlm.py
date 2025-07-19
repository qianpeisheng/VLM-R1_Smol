#!/usr/bin/env python3
"""
Test script for SmolVLM integration with VLM-R1
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

def test_integration():
    """Test integration with VLM-R1 framework"""
    try:
        # Test module selection
        sys.path.append('/home/peisheng/VLM-R1/src/open-r1-multimodal')
        from open_r1.grpo_jsonl import get_vlm_module
        
        # Test SmolVLM detection
        module_cls = get_vlm_module("HuggingFaceTB/SmolVLM-Instruct")
        print(f"✓ Module selection works: {module_cls.__name__}")
        
        # Test with different variations
        variations = [
            "SmolVLM-Instruct",
            "smolvlm-2b",
            "SmolVLM-2B-Instruct"
        ]
        
        for variation in variations:
            try:
                module_cls = get_vlm_module(variation)
                print(f"✓ Variation '{variation}' detected: {module_cls.__name__}")
            except Exception as e:
                print(f"⚠ Variation '{variation}' failed: {e}")
        
        print("\n✅ Integration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
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
        
        # Test IoU reward
        test_solution = [{"bbox": [100, 150, 200, 250]}, None, {"bbox": [40, 70, 160, 180]}]
        iou_rewards = SmolVLMModule.iou_reward(test_completions, test_solution)
        print(f"✓ IoU rewards: {iou_rewards}")
        
        print("\n✅ Reward function tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Reward function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Testing SmolVLM integration with VLM-R1...")
    print("=" * 50)
    
    success = True
    
    print("\n1. Testing SmolVLM Module...")
    success &= test_smolvlm_module()
    
    print("\n2. Testing Integration...")
    success &= test_integration()
    
    print("\n3. Testing Reward Functions...")
    success &= test_reward_functions()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! SmolVLM is ready for training.")
        print("\nNext steps:")
        print("1. Run the training script: ./run_scripts/run_grpo_rec_smolvlm_lora.sh")
        print("2. Monitor training with wandb")
        print("3. Evaluate the trained model")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)
