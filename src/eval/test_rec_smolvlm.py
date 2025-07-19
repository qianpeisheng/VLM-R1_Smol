#!/usr/bin/env python3
"""
Evaluation script for SmolVLM models on REC tasks
Based on the existing VLM-R1 evaluation scripts
"""

import sys
import os
import json
import torch
from tqdm import tqdm
import re
import argparse
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# Add VLM-R1 source to path
sys.path.append('/home/peisheng/VLM-R1/src/open-r1-multimodal')

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SmolVLM on REC tasks")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to the trained SmolVLM model")
    parser.add_argument("--data_root", type=str, 
                       default="/home/peisheng/VLM-R1/rec_jsons_processed",
                       help="Root directory for REC dataset JSONs")
    parser.add_argument("--image_root", type=str,
                       default="/home/peisheng/VLM-R1/coco",
                       help="Root directory for COCO images")
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                       help="Output directory for results")
    parser.add_argument("--datasets", type=str, nargs="+",
                       default=["refcoco_val", "refcocop_val", "refcocog_val"],
                       help="Datasets to evaluate on")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Batch size for evaluation")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to evaluate (for testing)")
    
    return parser.parse_args()

def load_model_and_processor(model_path):
    """Load SmolVLM model and processor"""
    try:
        from open_r1.vlm_modules.smolvlm_module import SmolVLMModule
        
        # Initialize SmolVLM module
        vlm_module = SmolVLMModule()
        
        # Load model
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
        
        # Post-init
        vlm_module.post_model_init(model, processor)
        
        print(f"✓ Loaded SmolVLM model from {model_path}")
        return model, processor, vlm_module
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
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

def format_prompt(question, vlm_module):
    """Format the REC prompt using SmolVLM's template"""
    template = vlm_module.get_question_template("rec")
    return template.replace("{Question}", question)

def parse_bbox_from_response(response):
    """Parse bounding box coordinates from model response"""
    # Pattern for [x1, y1, x2, y2]
    pattern = r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]"
    match = re.search(pattern, response)
    
    if match:
        return [int(x) for x in match.groups()]
    return None

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes"""
    if box1 is None or box2 is None:
        return 0.0
    
    # Calculate intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x1 >= x2 or y1 >= y2:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def evaluate_dataset(model, processor, vlm_module, dataset, image_root, args):
    """Evaluate model on a dataset"""
    model.eval()
    
    results = []
    correct_05 = 0
    correct_07 = 0
    format_correct = 0
    
    # Limit samples if specified
    if args.max_samples:
        dataset = dataset[:args.max_samples]
    
    print(f"Evaluating on {len(dataset)} samples...")
    
    for i, item in enumerate(tqdm(dataset, desc="Evaluating")):
        try:
            # Load image
            image_path = os.path.join(image_root, item["image"])
            if not os.path.exists(image_path):
                # Try different possible paths
                image_path = os.path.join(image_root, "train2014", item["image"])
                if not os.path.exists(image_path):
                    image_path = os.path.join(image_root, "val2014", item["image"])
            
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {item['image']}")
                continue
                
            image = Image.open(image_path).convert("RGB")
            
            # Format prompt
            question = item.get("prompt", item.get("question", ""))
            formatted_prompt = format_prompt(question, vlm_module)
            
            # Prepare inputs
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": formatted_prompt}
                ]}
            ]
            
            # Apply chat template
            prompt = processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Process inputs
            inputs = processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(model.device)
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
            
            # Decode response
            response = processor.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            # Parse predicted bbox
            pred_bbox = parse_bbox_from_response(response)
            gt_bbox = item.get("bbox", item.get("gt_bbox"))
            
            # Calculate metrics
            if pred_bbox is not None:
                format_correct += 1
                iou = calculate_iou(pred_bbox, gt_bbox)
                
                if iou >= 0.5:
                    correct_05 += 1
                if iou >= 0.7:
                    correct_07 += 1
            else:
                iou = 0.0
            
            # Store result
            result = {
                "image": item["image"],
                "question": question,
                "gt_bbox": gt_bbox,
                "pred_bbox": pred_bbox,
                "response": response,
                "iou": iou,
                "format_correct": pred_bbox is not None
            }
            results.append(result)
            
            # Log progress occasionally
            if (i + 1) % 100 == 0:
                acc_05 = correct_05 / (i + 1) * 100
                acc_07 = correct_07 / (i + 1) * 100
                format_acc = format_correct / (i + 1) * 100
                print(f"Progress: {i+1}/{len(dataset)} - "
                      f"Acc@0.5: {acc_05:.2f}% - "
                      f"Acc@0.7: {acc_07:.2f}% - "
                      f"Format: {format_acc:.2f}%")
        
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
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
        "accuracy_05": accuracy_05,
        "accuracy_07": accuracy_07,
        "format_accuracy": format_accuracy,
        "average_iou": avg_iou
    }
    
    return results, metrics

def main():
    args = parse_args()
    
    print("🚀 SmolVLM REC Evaluation")
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
            
            # Evaluate
            results, metrics = evaluate_dataset(
                model, processor, vlm_module, dataset, args.image_root, args
            )
            
            # Store results
            all_results[dataset_name] = results
            all_metrics[dataset_name] = metrics
            
            # Print metrics
            print(f"\n📈 Results for {dataset_name}:")
            print(f"  Total samples: {metrics['total_samples']}")
            print(f"  Accuracy@0.5: {metrics['accuracy_05']:.2f}%")
            print(f"  Accuracy@0.7: {metrics['accuracy_07']:.2f}%")
            print(f"  Format accuracy: {metrics['format_accuracy']:.2f}%")
            print(f"  Average IoU: {metrics['average_iou']:.4f}")
            
            # Save individual results
            output_file = os.path.join(args.output_dir, f"{dataset_name}_results.json")
            with open(output_file, 'w') as f:
                json.dump({
                    "metrics": metrics,
                    "results": results
                }, f, indent=2)
            print(f"  Results saved to: {output_file}")
            
        except Exception as e:
            print(f"❌ Failed to evaluate {dataset_name}: {e}")
            continue
    
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
    print("=" * 50)
    for dataset_name, metrics in all_metrics.items():
        print(f"{dataset_name:15} | "
              f"Acc@0.5: {metrics['accuracy_05']:6.2f}% | "
              f"Acc@0.7: {metrics['accuracy_07']:6.2f}% | "
              f"Format: {metrics['format_accuracy']:6.2f}%")
    
    print("\n🎉 Evaluation completed!")

if __name__ == "__main__":
    main()
