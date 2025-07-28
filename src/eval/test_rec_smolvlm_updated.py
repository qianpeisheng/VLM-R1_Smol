#!/usr/bin/env python3
"""
Updated SmolVLM REC Evaluation Script
Aligned with current VLM-R1 SmolVLM integration
"""

import sys
import os
import json
import torch
from tqdm import tqdm
import re
import argparse
from PIL import Image
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# Add VLM-R1 source to path
sys.path.append('/nas/xulei/VLM-R1_Smol/src/open-r1-multimodal')

from open_r1.vlm_modules.smolvlm_module import SmolVLMModule
from transformers import AutoProcessor, AutoModelForVision2Seq

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SmolVLM on REC tasks")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to the trained SmolVLM model")
    parser.add_argument("--data_root", type=str, 
                       default="/nas/xulei/VLM-R1_Smol/rec_jsons_processed",
                       help="Root directory for REC dataset JSONs")
    parser.add_argument("--image_root", type=str,
                       default="/nas/xulei/VLM-R1_Smol/coco",
                       help="Root directory for COCO images")
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                       help="Output directory for results")
    parser.add_argument("--datasets", type=str, nargs="+",
                       default=["refcoco_val", "refcocop_val", "refcocog_val"],
                       help="Datasets to evaluate on")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Batch size for evaluation")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to evaluate")
    parser.add_argument("--lisa_image_root", type=str,
                       default="/nas/xulei/VLM-R1_Smol/lisa",
                       help="Root directory for LISA images")
    
    return parser.parse_args()

def load_model_and_processor(model_path):
    """Load SmolVLM model with proper VLM-R1 integration"""
    try:
        # Initialize SmolVLM module (handles custom model loading)
        vlm_module = SmolVLMModule()
        
        # Load model with proper settings
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        )
        
        # Load processor
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # Apply post-init (important for custom model setup)
        vlm_module.post_model_init(model, processor)
        
        print(f"✓ Loaded SmolVLM model from {model_path}")
        return model, processor, vlm_module
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        raise

def load_dataset(data_root, dataset_name):
    """Load a REC dataset"""
    data_path = os.path.join(data_root, f"{dataset_name}.json")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    print(f"✓ Loaded {len(data)} samples from {dataset_name}")
    return data

def format_rec_prompt(question):
    """Format REC prompt - aligned with training format"""
    # Use the same template as in training
    template = "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
    return template.format(Question=question)

def extract_bbox_answer(content):
    """Extract bbox from model response - aligned with training format"""
    # Look for answer tags first (like in training)
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    # Look for JSON-like bbox format
    bbox_pattern = r'\{.*?\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\].*?\}'
    
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        bbox_match = re.search(bbox_pattern, content_answer, re.DOTALL)
        if bbox_match:
            bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), 
                   int(bbox_match.group(3)), int(bbox_match.group(4))]
            return bbox
    
    # Fallback: simple bracket pattern
    simple_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
    simple_match = re.search(simple_pattern, content)
    if simple_match:
        return [int(simple_match.group(1)), int(simple_match.group(2)), 
               int(simple_match.group(3)), int(simple_match.group(4))]
    
    return [0, 0, 0, 0]

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes"""
    if box1 is None or box2 is None or box1 == [0, 0, 0, 0]:
        return 0.0
    
    # Calculate intersection
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2]-1, box2[2]-1)
    inter_y2 = min(box1[3]-1, box2[3]-1)
    
    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        intersection = (inter_x2-inter_x1+1) * (inter_y2-inter_y1+1)
    else:
        intersection = 0
    
    # Calculate union
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - intersection
    
    return float(intersection) / union if union > 0 else 0.0

def evaluate_dataset(model, processor, vlm_module, dataset, image_root, args):
    """Evaluate model on a dataset using proper SmolVLM processing"""
    model.eval()
    
    results = []
    correct_05 = 0
    correct_07 = 0
    format_correct = 0
    
    # Limit samples if specified
    if args.max_samples:
        dataset = dataset[:args.max_samples]
    
    print(f"Evaluating on {len(dataset)} samples...")
    
    # Process in batches
    for i in tqdm(range(0, len(dataset), args.batch_size), desc="Evaluating"):
        batch_items = dataset[i:i + args.batch_size]
        
        try:
            # Prepare batch
            batch_images = []
            batch_prompts = []
            batch_valid_items = []
            
            for item in batch_items:
                # Load image
                image_path = os.path.join(image_root, item["image"])
                if not os.path.exists(image_path):
                    # Try LISA path if not found in COCO
                    image_path = os.path.join(args.lisa_image_root, item["image"])
                
                if not os.path.exists(image_path):
                    print(f"⚠️ Image not found: {item['image']}")
                    continue
                
                image = Image.open(image_path).convert("RGB")
                batch_images.append(image)
                
                # Format prompt with <image> token
                question = item.get("problem", item.get("question", ""))
                formatted_prompt = f"<image> {format_rec_prompt(question)}"
                batch_prompts.append(formatted_prompt)
                batch_valid_items.append(item)
            
            if not batch_images:
                continue
            
            # Use SmolVLM's custom input preparation
            model_inputs, _ = vlm_module.prepare_model_inputs(
                processor, 
                batch_prompts, 
                batch_images,
                return_tensors="pt",
                padding=True,
                padding_side="left"
            )
            
            # Move to device
            model_inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                          for k, v in model_inputs.items()}
            
            # Generate response with proper stopping criteria
            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    temperature=0.0,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    repetition_penalty=1.2,  # Add repetition penalty
                    no_repeat_ngram_size=3,  # Prevent 3-gram repetitions
                    use_cache=True
                )
            
            # Decode responses
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in 
                zip(model_inputs["input_ids"], generated_ids)
            ]
            
            batch_responses = processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            
            # Process results
            for j, (item, response) in enumerate(zip(batch_valid_items, batch_responses)):
                # Parse prediction
                predicted_bbox = extract_bbox_answer(response)
                ground_truth = item.get("solution", item.get("bbox", [0, 0, 0, 0]))
                
                # Calculate metrics
                iou_score = calculate_iou(predicted_bbox, ground_truth)
                correct_05 = correct_05 + 1 if iou_score > 0.5 else correct_05
                correct_07 = correct_07 + 1 if iou_score > 0.7 else correct_07
                format_correct = format_correct + 1 if predicted_bbox != [0, 0, 0, 0] else format_correct
                
                # Store result
                result = {
                    "image": item["image"],
                    "question": item.get("problem", item.get("question", "")),
                    "ground_truth": ground_truth,
                    "predicted_bbox": predicted_bbox,
                    "model_output": response,
                    "iou": iou_score,
                    "correct_05": iou_score > 0.5,
                    "correct_07": iou_score > 0.7
                }
                results.append(result)
                
        except Exception as e:
            print(f"❌ Error processing batch {i}: {e}")
            continue
    
    # Calculate final metrics
    total_samples = len(results)
    if total_samples > 0:
        accuracy_05 = correct_05 / total_samples * 100
        accuracy_07 = correct_07 / total_samples * 100
        format_accuracy = format_correct / total_samples * 100
        avg_iou = sum(r["iou"] for r in results) / total_samples
    else:
        accuracy_05 = accuracy_07 = format_accuracy = avg_iou = 0.0
    
    metrics = {
        "total_samples": total_samples,
        "accuracy_at_0.5": accuracy_05,
        "accuracy_at_0.7": accuracy_07,
        "format_accuracy": format_accuracy,
        "average_iou": avg_iou
    }
    
    return results, metrics

def main():
    args = parse_args()
    
    print("🚀 SmolVLM REC Evaluation (Updated)")
    print("=" * 50)
    print(f"Model: {args.model_path}")
    print(f"Datasets: {args.datasets}")
    print(f"Output: {args.output_dir}")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print("\n📥 Loading model...")
    model, processor, vlm_module = load_model_and_processor(args.model_path)
    
    # Evaluate on each dataset
    all_results = {}
    all_metrics = {}
    
    for dataset_name in args.datasets:
        print(f"\n📊 Evaluating on {dataset_name}...")
        
        try:
            # Load dataset
            dataset = load_dataset(args.data_root, dataset_name)
            
            # Choose correct image root
            if "lisa" in dataset_name.lower():
                current_image_root = args.lisa_image_root
            else:
                current_image_root = args.image_root
            
            # Evaluate
            results, metrics = evaluate_dataset(
                model, processor, vlm_module, dataset, current_image_root, args
            )
            
            # Store results
            all_results[dataset_name] = results
            all_metrics[dataset_name] = metrics
            
            # Save individual results
            result_file = os.path.join(args.output_dir, f"{dataset_name}_results.json")
            with open(result_file, 'w') as f:
                json.dump({
                    "dataset": dataset_name,
                    "metrics": metrics,
                    "results": results
                }, f, indent=2)
            
            print(f"📊 {dataset_name} Results:")
            print(f"   Acc@0.5: {metrics['accuracy_at_0.5']:.2f}%")
            print(f"   Acc@0.7: {metrics['accuracy_at_0.7']:.2f}%")
            print(f"   Format:  {metrics['format_accuracy']:.2f}%")
            print(f"   Avg IoU: {metrics['average_iou']:.4f}")
            
        except Exception as e:
            print(f"❌ Failed to evaluate {dataset_name}: {e}")
            all_metrics[dataset_name] = {"error": str(e)}
    
    # Save summary
    summary_file = os.path.join(args.output_dir, "evaluation_summary.json")
    with open(summary_file, 'w') as f:
        json.dump({
            "model_path": args.model_path,
            "datasets": args.datasets,
            "metrics": all_metrics
        }, f, indent=2)
    
    print(f"\n💾 Summary saved to: {summary_file}")
    
    # Print overall summary
    print("\n📋 Overall Summary:")
    print("=" * 70)
    for dataset_name, metrics in all_metrics.items():
        if "error" not in metrics:
            print(f"{dataset_name:15} | "
                  f"Acc@0.5: {metrics['accuracy_at_0.5']:6.2f}% | "
                  f"Acc@0.7: {metrics['accuracy_at_0.7']:6.2f}% | "
                  f"Format: {metrics['format_accuracy']:6.2f}%")
        else:
            print(f"{dataset_name:15} | ERROR: {metrics['error']}")
    
    print("\n🎉 Evaluation completed!")

if __name__ == "__main__":
    main()
