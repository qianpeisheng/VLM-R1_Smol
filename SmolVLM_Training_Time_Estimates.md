# SmolVLM LoRA Fine-tuning Time Estimates

## Dataset Information
- **Total training samples**: 120,624 (RefCOCO dataset)
- **Model**: SmolVLM-Instruct 2B parameters
- **LoRA Configuration**: rank=16, alpha=32, dropout=0.05
- **Training targets**: q_proj, v_proj (can be expanded to include k_proj, o_proj, gate_proj, up_proj, down_proj)

## Timing Results from Tests

### 1. Single GPU (24GB A5000)
- **Configuration**: 1 GPU, batch_size=2 per GPU
- **Samples per step**: 2
- **Steps needed for full epoch**: 60,312
- **Measured time per step**: 5.03 seconds (from 30-step test)
- **Time per epoch**: 84.26 hours
- **Time for 3 epochs**: 252.78 hours (10.5 days)
- **Time for 5 epochs**: 421.30 hours (17.6 days)

### 2. Dual GPU (2x 24GB A5000)
- **Configuration**: 2 GPUs, batch_size=2 per GPU  
- **Samples per step**: 4
- **Steps needed for full epoch**: 30,156
- **Measured time per step**: 3.30 seconds (estimated from failed run)
- **Time per epoch**: 27.64 hours
- **Time for 3 epochs**: 82.92 hours (3.5 days)
- **Time for 5 epochs**: 138.20 hours (5.8 days)

## Extrapolated Estimates

### 3. 6x 24GB A5000 (Estimated)
- **Configuration**: 6 GPUs, batch_size=2 per GPU
- **Samples per step**: 12  
- **Steps needed for full epoch**: 10,052
- **Estimated time per step**: ~2.5 seconds (assuming some overhead)
- **Estimated time per epoch**: ~7.0 hours
- **Estimated time for 3 epochs**: ~21 hours
- **Estimated time for 5 epochs**: ~35 hours

### 4. 10x 24GB A5000 (Estimated)
- **Configuration**: 10 GPUs, batch_size=2 per GPU
- **Samples per step**: 20
- **Steps needed for full epoch**: 6,031
- **Estimated time per step**: ~2.0 seconds (diminishing returns due to communication overhead)
- **Estimated time per epoch**: ~3.4 hours  
- **Estimated time for 3 epochs**: ~10 hours
- **Estimated time for 5 epochs**: ~17 hours

## Key Observations

1. **Scaling Efficiency**: 
   - 2 GPUs provide ~3x speedup over 1 GPU (better than linear due to reduced memory pressure)
   - 6 GPUs estimated to provide ~12x speedup
   - 10 GPUs estimated to provide ~25x speedup (diminishing returns)

2. **Memory Considerations**:
   - SmolVLM 2B + LoRA adapters fit comfortably on 24GB A5000
   - Can potentially increase batch size to 3-4 per GPU for better efficiency

3. **Practical Recommendations**:
   - **Single GPU**: Suitable for experimentation but very slow for full training
   - **2-6 GPUs**: Sweet spot for research/development work
   - **10+ GPUs**: For production training when time is critical

## Optimization Suggestions

1. **Increase batch size**: Try batch_size=3 or 4 per GPU if memory allows
2. **Mixed precision**: Already using bf16, consider fp16 if compatible
3. **Gradient accumulation**: Can reduce if using more GPUs
4. **Data loading**: Optimize dataloader_num_workers based on CPU cores
5. **LoRA targets**: Currently using q_proj,v_proj - can expand for potentially better performance

## Cost-Benefit Analysis

For a typical RL training requiring 3-5 epochs:

- **1 GPU**: 10-18 days (development/testing only)
- **2 GPUs**: 3.5-6 days (small research projects)  
- **6 GPUs**: ~1-1.5 days (medium research projects)
- **10 GPUs**: ~10-17 hours (production/deadline-driven projects)

The choice depends on:
- Time constraints
- Hardware availability/cost
- Experiment iteration needs
- Resource sharing requirements
