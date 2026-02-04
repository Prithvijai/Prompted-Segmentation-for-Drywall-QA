# Prompted-Segmentation-for-Drywall-QA

This repository contains the implementation of a text-conditioned segmentation system for construction automation. The project focuses on fine-tuning a foundation model to identify drywall cracks and taping joints to provide perception data for robotic navigation and repair tasks.

## Project Objective
The primary goal is to develop a model capable of outputting binary masks for "segment cracks" and "segment tapping area" given an image and a natural language prompt. This automation aims to address productivity challenges in the construction industry by bypassing manual inspection.

## Methodology
* **Core Model**: Fine-tuned SAM 3 (Segment Anything Model 3), a 0.85 Billion parameter foundation model released in November 2025.
* **PEFT Technique**: Utilized Low-Rank Adaptation (LoRA) to specialize the model while preserving foundational knowledge.
* **LoRA Configuration**: A rank of 16 was used, targeting Query, Value, Key, and Output projections in the self-attention layers.



## Dataset and Pre-processing
* **Dataset Composition**: Approximately 6,000 images combining Taping Area and Wall Crack data.
* **Data Sources**: Downloaded from Roboflow with an 80/10/10 split for training, validation, and testing.
* **Pre-processing**: Images were resized to 640x640. Taping area bounding boxes were converted to rectangular segmentation masks.

## Training and Results
Training was executed on the ASU Sol supercomputer using an NVIDIA A100 GPU.

### Training Parameters
| Parameter | Value |
| :--- | :--- |
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Loss Function | Binary Cross-Entropy (BCE) |
| Epochs | 30 |
| Batch Size | 8 |



### Performance Metrics
| Model Variant | Threshold | mIoU | Dice | Latency (ms) |
| :--- | :--- | :--- | :--- | :--- |
| Base SAM 3 | 0.05 | 0.0000 | 0.0000 | 393.04 |
| SAM 3 + LoRA | 0.50 | 0.3468 | 0.4736 | 425.79 |
| **Quantized (FP16)** | **0.50** | **0.3464** | **0.4732** | **142.80** |

The FP16 variant achieved a 66.5% reduction in latency compared to the full-precision LoRA model, making it suitable for robotics deployment.

## Repository Structure
* `train.py`: Main training execution script.
* `eval.py`: Performance evaluation and metric generation.
* `src/`: Modular codebase containing preprocessing, model definitions, and the training engine.