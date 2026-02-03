
import torch
from PIL import Image

from transformers import Sam3Processor, Sam3Model
from peft import LoraConfig, get_peft_model
# import matplotlib.pyplot as plt

def load_SAM3_LoRA_model(device, model_id="facebook/sam3", r=16, alpha=32, dtype=torch.float16):
    """Loads SAM3 + LoRA model
    
    load the model and processor of facebook/sam3 and configure LoRA needed to fine tune the model.
    """
    
    model = Sam3Model.from_pretrained(model_id, torch_dtype=dtype).to(device)
    processor = Sam3Processor.from_pretrained(model_id)

    config = LoraConfig(
        r = r,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
        lora_dropout=0.05,
        bias="none"

    )

    model = get_peft_model(model, config)

    return model, processor


