from open_r1.vlm_modules.vlm_module import VLMBaseModule
from typing import Dict, Any, Union, List, Optional
import torch
from PIL import Image
from transformers import (
    AutoProcessor, 
    AutoModelForVision2Seq,
    Idefics3ForConditionalGeneration,
    AutoConfig
)
import re
import os
import json
import sys
from datetime import datetime
from trl.data_utils import maybe_apply_chat_template

# Add the SmolVLM custom model path
sys.path.append('/nas/xulei/VLM-R1_Smol/smollm/vision/smolvlm2/smolvlm/model')
from modeling_smolvlm import SmolVLMForConditionalGeneration

class SmolVLMModule(VLMBaseModule):
    def __init__(self):
        super().__init__()
        
    def get_vlm_key(self):
        return "smolvlm"
    
    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        """SmolVLM uses custom SmolVLMForConditionalGeneration with fixed inputs_merger"""
        if "smolvlm" not in model_id.lower():
            raise ValueError(f"Expected SmolVLM model, got: {model_id}")
        
        # SmolVLM requires trust_remote_code
        model_init_kwargs["trust_remote_code"] = True
        
        # For LoRA compatibility, ensure we can load in specific dtype
        if "torch_dtype" not in model_init_kwargs:
            model_init_kwargs["torch_dtype"] = torch.bfloat16
        
        # Use the custom SmolVLM model with fixed inputs_merger
        return SmolVLMForConditionalGeneration
    
    def post_model_init(self, model, processing_class):
        """SmolVLM post-initialization"""
        return super().post_model_init(model, processing_class)
    
    def is_embeds_input(self):
        """SmolVLM uses input_ids with the custom model's fixed inputs_merger"""
        return False
    
    def get_processing_class(self):
        """SmolVLM uses AutoProcessor"""
        return AutoProcessor
    
    def get_vision_modules_keywords(self):
        """Keywords for freezing vision modules - important for LoRA"""
        # Based on SmolVLM/Idefics3 architecture
        return ['vision_model', 'connector']
    
    def get_custom_multimodal_keywords(self):
        """SmolVLM's multimodal inputs"""
        return ['pixel_values', 'pixel_attention_mask']
    
    def get_non_generate_params(self):
        """Parameters not used in generation - exclude pixel_attention_mask which may confuse generation"""
        return ['pixel_attention_mask']
    
    def get_custom_processing_keywords(self):
        """SmolVLM processing keywords"""
        return []
    
    def prepare_prompt(self, processing_class, inputs: dict[str, Union[torch.Tensor, Any]]):
        """Prepare prompts with SmolVLM's chat template"""
        prompts_text = []
        
        for i, example in enumerate(inputs):
            text_content = ""
            image_count = 0
            
            # Debug: Print the example structure for the first few
            if i < 2:
                print(f"SmolVLM Debug: Example {i} keys: {list(example.keys())}")
                if "prompt" in example:
                    print(f"SmolVLM Debug: Prompt type: {type(example['prompt'])}")
                    if isinstance(example["prompt"], list) and len(example["prompt"]) > 0:
                        print(f"SmolVLM Debug: First message: {example['prompt'][0]}")
                        if len(example["prompt"][0].get('content', [])) > 0:
                            print(f"SmolVLM Debug: Content items: {example['prompt'][0]['content']}")
            
            if "prompt" in example:
                if isinstance(example["prompt"], list):
                    # Conversational format
                    messages = example["prompt"]
                    
                    # Extract text content and count images
                    for msg in messages:
                        if isinstance(msg, dict) and "content" in msg:
                            content = msg["content"]
                            if isinstance(content, list):
                                # Extract text and count images from multimodal content
                                for item in content:
                                    if isinstance(item, dict):
                                        if item.get("type") == "image":
                                            image_count += 1
                                        elif item.get("type") == "text" and item.get("text"):
                                            text_content = item.get("text", "")
                            else:
                                # Simple string content
                                text_content = str(content)
                else:
                    # Simple string format
                    text_content = str(example["prompt"])
            else:
                # Fallback to 'text' field
                text_content = str(example.get("text", ""))
            
            # If we didn't find images in the prompt structure, check other fields
            if image_count == 0:
                # Check for image-related fields in the example
                if "image" in example:
                    if isinstance(example["image"], list):
                        image_count = len(example["image"])
                    else:
                        image_count = 1
                elif "image_path" in example:
                    if isinstance(example["image_path"], list):
                        image_count = len(example["image_path"])
                    else:
                        image_count = 1
            
            # Add image tokens at the beginning if we have images
            if image_count > 0:
                # SmolVLM expects <image> tokens at the start
                prompt = "<image>" * image_count + " " + text_content
            else:
                prompt = text_content
            
            prompts_text.append(prompt)
            if len(prompts_text) <= 2:  # Only print first few for debugging
                print(f"SmolVLM: Prepared prompt with {image_count} images: {prompt[:100]}...")
        
        return prompts_text
    
    def prepare_model_inputs(self, processing_class, prompts_text, images, 
                           return_tensors="pt", padding=True, 
                           padding_side="left", add_special_tokens=False):
        """Process prompts and images into model inputs for SmolVLM - Uses custom SmolVLM model with fixed inputs_merger"""
        
        print(f"SmolVLM Debug: prepare_model_inputs called with custom model:")
        print(f"  - Number of prompts: {len(prompts_text)}")
        print(f"  - Number of images: {len(images)}")
        print(f"  - First prompt: {prompts_text[0][:100] if prompts_text else 'None'}...")
        
        if len(images) == 0:
            # Text-only case - use tokenizer directly
            model_inputs = processing_class.tokenizer(
                prompts_text,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens
            )
            print("SmolVLM: Processed text-only inputs")
            return model_inputs, None
        
        # For multimodal inputs, we need to ensure correct pairing
        total_image_tokens = sum(prompt.count('<image>') for prompt in prompts_text)
        print(f"SmolVLM: Total <image> tokens: {total_image_tokens}, Available images: {len(images)}")
        
        # Handle different cases
        if total_image_tokens == len(images):
            pass  # Perfect match
        elif total_image_tokens == 0 and len(images) > 0:
            # No image tokens but we have images - add tokens to the beginning of prompts
            print("SmolVLM: Adding missing <image> tokens to prompts")
            prompts_text = [f"<image> {prompt}" for prompt in prompts_text[:len(images)]] + prompts_text[len(images):]
        elif total_image_tokens > len(images):
            # More tokens than images - duplicate images
            print("SmolVLM: Duplicating images to match tokens")
            while len(images) < total_image_tokens:
                images.append(images[-1])
        elif total_image_tokens < len(images):
            # More images than tokens - use only needed images
            print("SmolVLM: Using subset of images")
            images = images[:total_image_tokens]
        
        try:
            # Process with the corrected inputs - the custom SmolVLM model should handle inputs_merger correctly
            model_inputs = processing_class(
                text=prompts_text,
                images=images,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens
            )
            
            print(f"SmolVLM: Successfully processed inputs with custom model")
            print(f"SmolVLM: Keys: {list(model_inputs.keys())}")
            if 'input_ids' in model_inputs:
                print(f"SmolVLM: Input IDs shape: {model_inputs['input_ids'].shape}")
                # Count token types for debugging
                fake_token_id = 49152  # <fake_token_around_image>
                image_token_id = 49153  # <image>
                real_image_tokens = (model_inputs['input_ids'] == image_token_id).sum().item()
                fake_tokens = (model_inputs['input_ids'] == fake_token_id).sum().item()
                print(f"SmolVLM: Real image tokens: {real_image_tokens}, Fake tokens: {fake_tokens}")
            if 'pixel_values' in model_inputs:
                print(f"SmolVLM: Pixel values shape: {model_inputs['pixel_values'].shape}")
            
            return model_inputs, None
            
        except Exception as e:
            print(f"SmolVLM Error: {e}")
            import traceback
            traceback.print_exc()
            # Emergency fallback to text-only
            model_inputs = processing_class.tokenizer(
                prompts_text,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens
            )
            print("SmolVLM: Fell back to text-only processing")
            return model_inputs, None
    
    @staticmethod
    def get_question_template(task_type: str):
        """Return question template for specific tasks"""
        if task_type == "rec":
            return "{Question} Please provide the bounding box coordinates as [x1, y1, x2, y2] where the values are integers between 0 and 1000."
        elif task_type == "ovd":
            return "Detect all {Question} in the image and provide their bounding boxes as [x1, y1, x2, y2]."
        else:
            return "{Question}"
    
    @staticmethod
    def format_reward_rec(completions, **kwargs):
        """Check if output matches REC/OVD format"""
        # Pattern for bounding box: [x1, y1, x2, y2]
        pattern = r"\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]"
        
        completion_contents = []
        for completion in completions:
            if isinstance(completion, list) and len(completion) > 0:
                content = completion[0].get("content", "")
            else:
                content = str(completion)
            completion_contents.append(content)
        
        matches = [re.search(pattern, content) is not None for content in completion_contents]
        
        # Debug logging
        if os.getenv("DEBUG_MODE") == "true":
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            log_path = os.getenv("LOG_PATH", "debug_smolvlm.txt")
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Format reward -------------\n")
                for content, match in zip(completion_contents, matches):
                    f.write(f"Content: {content}\n")
                    f.write(f"Has format: {bool(match)}\n")
        
        return [1.0 if match else 0.0 for match in matches]
    
    @staticmethod
    def iou_reward(completions, solution, **kwargs):
        """Calculate IoU reward - reusing VLM-R1's implementation"""
        import re
        import numpy as np
        
        def parse_bbox(text):
            """Extract bounding box from text"""
            pattern = r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]"
            match = re.search(pattern, text)
            if match:
                return [int(x) for x in match.groups()]
            return None
        
        def calculate_iou(box1, box2):
            """Calculate IoU between two boxes"""
            # Ensure boxes are numeric - convert strings to floats if needed
            def ensure_numeric(box):
                if box is None:
                    return None
                # Handle different input formats
                if isinstance(box, (list, tuple)):
                    try:
                        return [float(x) for x in box]
                    except (ValueError, TypeError):
                        return None
                elif isinstance(box, str):
                    # Try to parse as bbox string
                    parsed = parse_bbox(box)
                    if parsed:
                        return [float(x) for x in parsed]
                    return None
                else:
                    return None
            
            box1 = ensure_numeric(box1)
            box2 = ensure_numeric(box2)
            
            if box1 is None or box2 is None:
                return 0.0
            
            # Ensure we have 4 coordinates each
            if len(box1) != 4 or len(box2) != 4:
                return 0.0
            
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0
        
        rewards = []
        
        # Debug: print the inputs to understand the data format
        print(f"SmolVLM IoU Debug: Processing {len(completions)} completions")
        print(f"SmolVLM IoU Debug: Solution type: {type(solution)}")
        if isinstance(solution, list) and len(solution) > 0:
            print(f"SmolVLM IoU Debug: First solution type: {type(solution[0])}, value: {solution[0]}")
        
        # Handle both list and single solution formats
        if not isinstance(solution, list):
            solution = [solution] * len(completions)
        
        for i, completion in enumerate(completions):
            if isinstance(completion, list) and len(completion) > 0:
                content = completion[0].get("content", "")
            else:
                content = str(completion)
            
            pred_bbox = parse_bbox(content)
            
            # Debug: print what we found
            print(f"SmolVLM IoU Debug: Completion {i}: content='{content[:100]}...'")
            print(f"SmolVLM IoU Debug: Parsed bbox: {pred_bbox}")
            
            # Handle solution format
            if i < len(solution) and solution[i] is not None:
                sol = solution[i]
                if isinstance(sol, dict):
                    gt_bbox = sol.get("bbox", sol.get("gt_bbox", None))
                else:
                    gt_bbox = sol
            else:
                gt_bbox = None
            
            print(f"SmolVLM IoU Debug: GT bbox type: {type(gt_bbox)}, value: {gt_bbox}")
            
            if pred_bbox and gt_bbox:
                iou = calculate_iou(pred_bbox, gt_bbox)
                # Soft IoU reward as in VLM-R1
                reward = 2 * iou - 1
            else:
                reward = -1.0
            
            rewards.append(reward)
        
        return rewards
    
    @staticmethod
    def select_reward_func(func: str, task_type: str):
        """Select appropriate reward function"""
        if func == "accuracy":
            if task_type in ["rec", "ovd"]:
                return SmolVLMModule.iou_reward
            else:
                raise ValueError(f"Unsupported task type for accuracy: {task_type}")
        elif func == "format":
            if task_type in ["rec", "ovd"]:
                return SmolVLMModule.format_reward_rec
            else:
                raise ValueError(f"Unsupported task type for format: {task_type}")
        else:
            # Delegate to base reward functions in grpo_jsonl.py
            return None
