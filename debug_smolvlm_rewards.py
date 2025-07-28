#!/usr/bin/env python3
"""
Debug script for SmolVLM IoU reward calculation
This script helps debug the training data format and reward calculation
"""

import sys
import os
import json
sys.path.append('/nas/xulei/VLM-R1_Smol/src/open-r1-multimodal')

from open_r1.vlm_modules.smolvlm_module import SmolVLMModule

def test_iou_reward_calculation():
    """Test the IoU reward calculation with sample data"""
    
    print("🔍 Testing SmolVLM IoU Reward Calculation")
    print("=" * 50)
    
    # Test cases with different formats
    test_cases = [
        {
            "name": "Perfect match with answer tags",
            "completion": [{"content": "<think>I need to find the person</think><answer>[100, 200, 300, 400]</answer>"}],
            "solution": [100, 200, 300, 400],
            "expected_iou": 1.0,
            "expected_reward": 1.0
        },
        {
            "name": "Partial overlap",
            "completion": [{"content": "The bounding box is [100, 200, 250, 350]"}],
            "solution": [100, 200, 300, 400],
            "expected_iou": 0.375,  # Calculated manually
            "expected_reward": -0.25  # 2 * 0.375 - 1
        },
        {
            "name": "No bbox in output",
            "completion": [{"content": "I can see a person but cannot determine coordinates"}],
            "solution": [100, 200, 300, 400],
            "expected_iou": 0.0,
            "expected_reward": -1.0
        },
        {
            "name": "JSON format solution",
            "completion": [{"content": 'Answer: {"bbox": [150, 250, 350, 450]}'}],
            "solution": {"bbox": [100, 200, 300, 400]},
            "expected_iou": 0.36,  # Approximate
            "expected_reward": -0.28  # 2 * 0.36 - 1
        },
        {
            "name": "String solution format",
            "completion": [{"content": "The answer is [100, 200, 300, 400]"}],
            "solution": "<answer>[100, 200, 300, 400]</answer>",
            "expected_iou": 1.0,
            "expected_reward": 1.0
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n📋 Test Case {i+1}: {test_case['name']}")
        print("-" * 40)
        
        # Calculate reward
        rewards = SmolVLMModule.iou_reward(
            [test_case["completion"]], 
            [test_case["solution"]]
        )
        
        actual_reward = rewards[0]
        expected_reward = test_case["expected_reward"]
        
        print(f"Completion: {test_case['completion'][0]['content'][:100]}...")
        print(f"Solution: {test_case['solution']}")
        print(f"Expected reward: {expected_reward:.3f}")
        print(f"Actual reward: {actual_reward:.3f}")
        
        # Check if close enough (allow small floating point differences)
        if abs(actual_reward - expected_reward) < 0.05:
            print("✅ PASS")
        else:
            print("❌ FAIL")

def test_format_reward():
    """Test the format reward calculation"""
    
    print("\n\n🔍 Testing SmolVLM Format Reward")
    print("=" * 50)
    
    test_cases = [
        {
            "name": "Valid answer tags format",
            "completion": [{"content": "<think>Finding object</think><answer>[100, 200, 300, 400]</answer>"}],
            "expected": 1.0
        },
        {
            "name": "Valid simple bracket format",
            "completion": [{"content": "The bounding box is [100, 200, 300, 400]"}],
            "expected": 1.0
        },
        {
            "name": "Valid JSON format",
            "completion": [{"content": '{"bbox": [100, 200, 300, 400]}'}],
            "expected": 1.0
        },
        {
            "name": "Invalid format - no bbox",
            "completion": [{"content": "I can see a person in the image"}],
            "expected": 0.0
        },
        {
            "name": "Invalid format - incomplete bbox",
            "completion": [{"content": "The coordinates are [100, 200]"}],
            "expected": 0.0
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n📋 Test Case {i+1}: {test_case['name']}")
        print("-" * 40)
        
        rewards = SmolVLMModule.format_reward_rec([test_case["completion"]])
        actual_reward = rewards[0]
        expected_reward = test_case["expected"]
        
        print(f"Completion: {test_case['completion'][0]['content'][:100]}...")
        print(f"Expected reward: {expected_reward}")
        print(f"Actual reward: {actual_reward}")
        
        if actual_reward == expected_reward:
            print("✅ PASS")
        else:
            print("❌ FAIL")

def check_training_data_format():
    """Check the actual training data format"""
    
    print("\n\n🔍 Checking Training Data Format")
    print("=" * 50)
    
    data_files = [
        "/nas/xulei/VLM-R1_Smol/rec_jsons_processed/refcoco_train.json",
        "/nas/xulei/VLM-R1_Smol/rec_jsons_processed/refcoco_val.json"
    ]
    
    for data_file in data_files:
        if os.path.exists(data_file):
            print(f"\n📁 Checking {os.path.basename(data_file)}")
            print("-" * 30)
            
            with open(data_file, 'r') as f:
                data = json.load(f)
            
            print(f"Total samples: {len(data)}")
            
            # Check first few samples
            for i in range(min(3, len(data))):
                sample = data[i]
                print(f"\nSample {i+1}:")
                print(f"  Keys: {list(sample.keys())}")
                
                if 'problem' in sample:
                    print(f"  Problem: {sample['problem'][:100]}...")
                elif 'question' in sample:
                    print(f"  Question: {sample['question'][:100]}...")
                
                if 'solution' in sample:
                    solution = sample['solution']
                    print(f"  Solution type: {type(solution)}")
                    print(f"  Solution: {solution}")
                elif 'bbox' in sample:
                    print(f"  Bbox: {sample['bbox']}")
                
                if 'image' in sample:
                    print(f"  Image: {sample['image']}")
        else:
            print(f"❌ File not found: {data_file}")

def main():
    """Main function"""
    # Enable debug mode for detailed logging
    os.environ["DEBUG_MODE"] = "true"
    os.environ["LOG_PATH"] = "/nas/xulei/VLM-R1_Smol/debug_reward_test.log"
    
    print("🚀 SmolVLM Reward Debug Script")
    print("=" * 60)
    
    # Test IoU reward calculation
    test_iou_reward_calculation()
    
    # Test format reward calculation
    test_format_reward()
    
    # Check actual training data format
    check_training_data_format()
    
    print("\n\n✅ Debug script completed!")
    print(f"📄 Detailed logs saved to: {os.environ['LOG_PATH']}")

if __name__ == "__main__":
    main()
