
import torch 
from torch import nn

from src.preprocessing import DrywallQADatasetCustom
from src.model import load_SAM3_LoRA_model
from torch.utils.data import ConcatDataset, DataLoader
from src.utils import collate_fn
from src.engine import train_model


device = "cuda" if torch.cuda.is_available() else "cpu"

model, processor = load_SAM3_LoRA_model(device=device)
model.print_trainable_parameters()


d1_train = DrywallQADatasetCustom("./datasets/Drywall_Join_Detect_v1/train", prompt="segment taping area")
d2_train = DrywallQADatasetCustom("./datasets/cracks_v1/train", prompt="segment wall crack")

train_ds = ConcatDataset([d1_train, d2_train])
train_loader = DataLoader(
    train_ds,
    batch_size=1,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn
)

d1_vaild = DrywallQADatasetCustom("./datasets/Drywall_Join_Detect_v1/valid", prompt="segment taping area")
d2_vaild = DrywallQADatasetCustom("./datasets/cracks_v1/valid", prompt="segment wall crack")

val_ds = ConcatDataset([d1_vaild, d2_vaild])
val_loader = DataLoader(
    val_ds,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.BCEWithLogitsLoss()

scaler = torch.amp.GradScaler()



train_model(model, processor, train_loader, val_loader, optimizer, loss_fn, epochs=2, device=device)

