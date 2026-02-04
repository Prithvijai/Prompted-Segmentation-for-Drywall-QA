
import torch 
from torch import nn
import os

from src.preprocessing import DrywallQADatasetCustom
from src.model import load_SAM3_LoRA_model
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler
from src.utils import collate_fn
from src.engine import train_model


device = "cuda" if torch.cuda.is_available() else "cpu"

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

model, processor = load_SAM3_LoRA_model(device=device,dtype=torch.float16)
model.print_trainable_parameters()


d1_train = DrywallQADatasetCustom("./datasets/Drywall_Join_Detect_v1/train", prompt="segment taping area")
d2_train = DrywallQADatasetCustom("./datasets/cracks_v1/train", prompt="segment wall crack")

train_ds = ConcatDataset([d1_train, d2_train])

train_ds_s = RandomSampler(
    train_ds, 
    replacement=False, 
    num_samples=1000
)

train_loader = DataLoader(
    train_ds,
    batch_size=8,
    # shuffle=True,
    sampler=train_ds_s,
    num_workers=12,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=collate_fn
)

d1_vaild = DrywallQADatasetCustom("./datasets/Drywall_Join_Detect_v1/valid", prompt="segment taping area")
d2_vaild = DrywallQADatasetCustom("./datasets/cracks_v1/valid", prompt="segment wall crack")

val_ds = ConcatDataset([d1_vaild, d2_vaild])
val_loader = DataLoader(
    val_ds,
    batch_size=8,
    shuffle=False,
    num_workers=12,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=collate_fn
)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.BCEWithLogitsLoss()

scaler = torch.amp.GradScaler()


train_model(model, processor, train_loader, val_loader, optimizer, loss_fn, epochs=30, device=device)

