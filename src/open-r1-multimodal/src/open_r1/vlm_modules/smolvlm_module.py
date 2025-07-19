from open_r1.vlm_modules.vlm_module import VLMBaseModule
from typing import Dict, Any, Union, List, Optional
import torch
from PIL import Image
from transformers import (
    AutoProcessor, 
    AutoModelForVision2Seq,
    AutoConfig
)
import re
import os
import json
from datetime import datetime
from trl.data_utils import maybe_apply_chat_template

class SmolVLMModule(VLMBaseModule):
    def __init__(self):
        super().__init__()
        
    def get_vlm_key(self):
        return "smolvlm"
    
    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        """SmolVLM uses AutoModelForVision2Seq"""
        if "smolvlm" not in model_id.lower():
            raise ValueError(f"Expected SmolVLM model, got: {model_id}")
        
        # SmolVLM requires trust_remote_code
        model_init_kwargs["trust_remote_code"] = True
        
        # For LoRA compatibility, ensure we can load in specific dtype
        if "torch_dtype" not in model_init_kwargs:
            model_init_kwargs["torch_dtype"] = torch.bfloat16
        
        return AutoModelForVision2Seq
    
    def post_model_init(self, model, processing_class):
        """SmolVLM post-initialization"""
        # Store processor for later use
        self.processor = processing_class
        
        # Check if model has special tokens we need to handle
        if hasattr(processing_class, 'image_token'):
            self.image_token = processing_class.image_token
        elif hasattr(processing_class.tokenizer, 'image_token'):
            self.image_token = processing_class.tokenizer.image_token
        else:
            self.image_token = "<image>"  # Default for SmolVLM
        
        print(f"SmolVLM: Using image token: {self.image_token}")
    
    def is_embeds_input(self):
        """SmolVLM uses input_ids"""
        return False
    
    def get_processing_class(self):
        """SmolVLM uses AutoProcessor"""
        return AutoProcessor
    
    def get_vision_modules_keywords(self):
        """Keywords for freezing vision modules - important for LoRA"""
        # Based on SmolVLM/Idefics3 architecture
        return ['vision_model', 'vision_encoder', 'image_encoder', 'visual_encoder', 'connector']
    
    def get_custom_multimodal_keywords(self):
        """SmolVLM's multimodal inputs"""
        return ['pixel_values']
    
    def get_non_generate_params(self):
        """Parameters not used in generation"""
        return []
    
    def get_custom_processing_keywords(self):
        """SmolVLM processing keywords"""
        # SmolVLM might have max_image_size or similar
        return []
    
    def prepare_prompt(self, processing_class, inputs: dict[str, Union[torch.Tensor, Any]]):
        """Prepare prompts with SmolVLM's chat template"""
        prompts_text = []
        
        for example in inputs:
            # Handle different input formats
            if "prompt" in example:
                if isinstance(example["prompt"], list):
                    # Conversational format - handle carefully for SmolVLM
                    messages = example["prompt"]
                    
                    # For SmolVLM, we need to handle image placement differently
                    # Let's use a simpler approach that works with the model's expectations
                    if messages and isinstance(messages[-1], dict):
                        user_content = messages[-1].get("content", "")
                        # Don't apply chat template if we have images - it might interfere
                        if "image" in example:
                            prompt = user_content
                        else:
                            try:
                                prompt = processing_class.apply_chat_template(
                                    messages, 
                                    tokenize=False, 
                                    add_generation_prompt=True
                                )
                            except Exception as e:
                                print(f"Chat template error: {e}, using simple format")
                                prompt = user_content
                    else:
                        prompt = str(example["prompt"])
                else:
                    # Simple string format
                    prompt = str(example["prompt"])
            else:
                # Fallback
                prompt = str(example.get("text", ""))
            
            # For SmolVLM, don't manually add image tokens - let the processor handle it
            # The processor will add them in the right places during processing
            
            prompts_text.append(prompt)
            print(f"SmolVLM: Prepared prompt: {prompt[:100]}...")
        
        return prompts_text
    
    def prepare_model_inputs(self, processing_class, prompts_text, images, 
                           return_tensors="pt", padding=True, 
                           padding_side="left", add_special_tokens=False):
        """Process prompts and images into model inputs"""
        
        additional_output = None
        
        try:
            if len(images) > 0:
                # For SmolVLM, let the processor handle image token insertion
                # Don't use add_special_tokens=False as it might interfere
                model_inputs = processing_class(
                    text=prompts_text,
                    images=images,
                    return_tensors=return_tensors,
                    padding=padding,
                    truncation=True,
                    max_length=2048,  # Increase max length to accommodate image tokens
                    # Let the processor decide on special tokens for images
                )
                
                # Debug print to understand what's happening
                print(f"SmolVLM: Processed {len(images)} images with {len(prompts_text)} prompts")
                if hasattr(model_inputs, 'pixel_values') and model_inputs.pixel_values is not None:
                    print(f"SmolVLM: pixel_values shape: {model_inputs.pixel_values.shape}")
                if hasattr(model_inputs, 'input_ids'):
                    print(f"SmolVLM: input_ids shape: {model_inputs.input_ids.shape}")
                    # Check for image token presence
                    if hasattr(processing_class.tokenizer, 'image_token_id'):
                        image_token_count = (model_inputs.input_ids == processing_class.tokenizer.image_token_id).sum().item()
                        print(f"SmolVLM: Found {image_token_count} image tokens in input_ids")
                
            else:
                # Process text-only
                model_inputs = processing_class.tokenizer(
                    prompts_text,
                    return_tensors=return_tensors,
                    padding=padding,
                    truncation=True,
                    max_length=2048,
                    add_special_tokens=True  # Use True for text-only
                )
                print(f"SmolVLM: Text-only processing for {len(prompts_text)} prompts")
        
        except Exception as e:
            print(f"SmolVLM processing error: {e}")
            # Fallback processing
            model_inputs = processing_class.tokenizer(
                prompts_text,
                return_tensors=return_tensors,
                padding=padding,
                truncation=True,
                max_length=2048,
                add_special_tokens=True
            )
        
        return model_inputs, additional_output
    
    # Task-specific methods for OVD/REC
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
