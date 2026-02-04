
import torch
from PIL import Image
import time
import numpy as np

from transformers import Sam3Processor, Sam3Model
from src.preprocessing import DrywallQADatasetCustom
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
from src.utils import collate_fn, calculate_metrics
from peft import PeftModel
from transformers import Sam3Processor, Sam3Model

from transformers import BitsAndBytesConfig

def evaluate_model(model, processor, dataloader, device, model_name, threshold=0.50):
    model.eval()
    total_iou = 0
    total_dice = 0
    inference_times = []
    
    print(f"\n--- Evaluating: {model_name} ---")
    
    with torch.inference_mode():
        for batch in tqdm(dataloader):
            images = batch["image"]
            gt_masks = batch["mask"].to(device) 
            prompts = batch["prompt"]
            
            start_time = time.perf_counter()
            inputs = processor(images, text=prompts, return_tensors="pt").to(device)
            outputs = model(**inputs)

            result = processor.post_process_instance_segmentation(
                outputs,
                threshold=threshold,      
                mask_threshold=0.5,    
                target_sizes=[images[0].size()[:2]] 
            )
            
            pred_masks = torch.nn.functional.interpolate(
                outputs.pred_masks.mean(dim=1, keepdim=True),
                size=gt_masks.shape[-2:],
                mode="bilinear"
            ).sigmoid()
            
            torch.cuda.synchronize() 
            inference_times.append(time.perf_counter() - start_time)
            
            for i in range(pred_masks.shape[0]):
                iou, dice = calculate_metrics(
                    pred_masks[i].squeeze().cpu().numpy(),
                    gt_masks[i].cpu().numpy()
                )
                total_iou += iou
                total_dice += dice

    avg_iou = total_iou / len(dataloader.dataset)
    avg_dice = total_dice / len(dataloader.dataset)
    avg_time = np.mean(inference_times) * 1000 
    
    return {"mIoU": avg_iou, "Dice": avg_dice, "Latency (ms)": avg_time}

def generate_final_report_visuals(models, processor, dataset, device):
    # index 5: Typically from Taping Area (Dataset 1)
    # index 500: Typically from Cracks (Dataset 2) - Adjust based on your split
    indices = [10, 503] 
    num_samples = len(indices)
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 10))
    model_keys = ["base", "lora", "quant"]
    model_labels = ["Base SAM3", "SAM3 + LoRA", "FP16 LoRA"]

    for row, img_idx in enumerate(indices):
        sample = dataset[img_idx]
        raw_image = sample['image']
        # FIX: Convert PIL Image to numpy array for plotting
        gt_mask = np.array(sample['mask']) 
        prompt = sample['prompt']
        
        # Column 0: Ground Truth (Green)
        axes[row, 0].imshow(raw_image)
        gt_overlay = np.ma.masked_where(gt_mask == 0, gt_mask)
        axes[row, 0].imshow(gt_overlay, cmap='Greens', alpha=0.6)
        if row == 0: axes[row, 0].set_title("Ground Truth", fontsize=14, fontweight='bold')
        axes[row, 0].set_ylabel(f"{prompt.split()[-1].capitalize()} Sample", fontsize=12)
        axes[row, 0].axis('off')

        # Columns 1-3: Model Predictions (Spring/Pink)
        for col, key in enumerate(model_keys):
            current_model = models[key]
            current_model.eval()
            inputs = processor(raw_image, text=prompt, return_tensors="pt").to(device)
            with torch.inference_mode():
                outputs = current_model(**inputs)
            
            thresh = 0.1 if key == "base" else 0.5
            pred_mask = torch.nn.functional.interpolate(
                outputs.pred_masks.mean(dim=1, keepdim=True),
                size=raw_image.size[::-1],
                mode="bilinear"
            ).sigmoid().cpu().numpy()[0][0]
            
            axes[row, col+1].imshow(raw_image)
            pred_overlay = np.ma.masked_where(pred_mask < thresh, pred_mask)
            axes[row, col+1].imshow(pred_overlay, cmap='spring', alpha=0.5)
            
            if row == 0:
                axes[row, col+1].set_title(model_labels[col], fontsize=14, fontweight='bold')
            axes[row, col+1].axis('off')

    plt.tight_layout()
    plt.savefig("drywall_qa_samples.png", dpi=300)
    print("\n[SUCCESS] Visuals for Taping and Cracks saved to drywall_qa_samples.png")


device = "cuda" if torch.cuda.is_available() else "cpu"


d1_test = DrywallQADatasetCustom("./datasets/Drywall_Join_Detect_v1/test", "segment taping area")
d2_test = DrywallQADatasetCustom("./datasets/cracks_v1/test", "segment wall crack")
test_ds = ConcatDataset([d1_test, d2_test])
test_loader = DataLoader(test_ds, batch_size=1, collate_fn=collate_fn) 


base_model = Sam3Model.from_pretrained("facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")
# results_base = evaluate_model(base_model, processor, test_loader, device, "Base SAM3", threshold=0.05)
# print(f"Base SAM3 Results: mIoU={results_base['mIoU']:.4f}, Dice={results_base['Dice']:.4f}, Latency={results_base['Latency (ms)']:.2f}ms")


lora_model = PeftModel.from_pretrained(base_model, "./models/sam3_lora_best_30epochs").to(device)
# results_lora = evaluate_model(lora_model, processor, test_loader, device, "SAM3 + LoRA", threshold=0.50)
# print(f"SAM3 + LoRA Results: mIoU={results_lora['mIoU']:.4f}, Dice={results_lora['Dice']:.4f}, Latency={results_lora['Latency (ms)']:.2f}ms")

# bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)

quant_model = Sam3Model.from_pretrained(
    "facebook/sam3", 
    torch_dtype=torch.float16
).to(device)

quant_lora = PeftModel.from_pretrained(
    quant_model, 
    "./models/sam3_lora_best_30epochs"
).to(device)

# results_quant = evaluate_model(quant_lora, processor, test_loader, device, "Quantized SAM3 + LoRA (FP16)", threshold=0.50)
# print(f"FP16 Results: mIoU={results_quant['mIoU']:.4f}, Dice={results_quant['Dice']:.4f}, Latency={results_quant['Latency (ms)']:.2f}ms")



# print("\nFinal Results Table:")
# print(f"{'Model':<25} | {'mIoU':<8} | {'Dice':<8} | {'Latency':<10}")
# for name, res in zip(["Base", "LoRA", "Quantized LoRA"], [results_base, results_lora, results_quant]):
#     print(f"{name:<25} | {res['mIoU']:.4f} | {res['Dice']:.4f} | {res['Latency (ms)']:.2f}ms")


model_dict = {
    "base": base_model,
    "lora": lora_model,
    "quant": quant_lora
}
generate_final_report_visuals(model_dict, processor, test_ds, device)