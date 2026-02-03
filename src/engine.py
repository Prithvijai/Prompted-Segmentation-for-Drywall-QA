
import torch
from tqdm import tqdm



scaler = torch.amp.GradScaler()

def train_step(model, processor, train_dataloader, optimizer, loss_fn, device):
    model.train()
    train_loss = 0
    pbar = tqdm(train_dataloader, desc="Training")

    for batch in pbar:
        masks = batch["mask"].to(device).unsqueeze(1)
        inputs = processor(batch['image'], text=batch["prompt"], return_tensors="pt").to(device)

        with torch.amp.autocast(device_type=device):
            outputs = model(**inputs)
            loss = loss_fn(outputs.pred_masks, masks)
            

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

    train_loss = train_loss /len(train_dataloader)

    return train_loss


def val_step(model, processor, test_dataloader, loss_fn, device):
    model.eval()
    val_loss = 0
    pbar = tqdm(test_dataloader, desc="Validation")


    with torch.inference_mode():
        for batch in pbar:
            masks = batch["mask"].to(device).unsqueeze(1)
            inputs = processor(batch['image'], text=batch["prompt"], return_tensors="pt").to(device)

            with torch.amp.autocast(device_type=device):
                outputs = model(**inputs)
                loss = loss_fn(outputs.pred_masks, masks)
                val_loss +=loss.item()

    val_loss = val_loss /len(test_dataloader)

    return val_loss


def train_model(model, processor, train_dataloader, test_dataloader, optimizer, loss_fn, epochs, device):
    best_val_loss = float("inf")
    for epoch in range(epochs):

        avg_train_loss = train_step(model, processor, train_dataloader, optimizer, loss_fn, device)
        avg_val_loss = val_step(model, processor, test_dataloader, optimizer, loss_fn, device)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained("./models/sam3_lora_best")
            print(f"--- New Best Model Saved (Val Loss: {best_val_loss:.4f}) ---")

        model.save_pretrained("./models/sam3_lora_final")
        print("Training Complete!")



