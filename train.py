import os
import csv
import time
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
from metrics import iou, dice_coeff, precision_recall, specificity, accuracy


def train_one_epoch(model, loader, criterion, optimizer, device, apply_sigmoid=True):
    model.train()
    total_loss = 0.0
    total_acc  = 0.0
    start_time = time.time()

    for batch in loader:
        images = batch["image"].to(device)
        masks  = batch["mask"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc  += accuracy(outputs, masks, apply_sigmoid=apply_sigmoid).item()

    epoch_time = time.time() - start_time
    return total_loss / len(loader), total_acc / len(loader), epoch_time


def validate_one_epoch(model, loader, criterion, device, apply_sigmoid=True):
    model.eval()
    total_loss = 0.0
    total_acc  = 0.0
    start_time = time.time()

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks  = batch["mask"].to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            total_loss += loss.item()
            total_acc  += accuracy(outputs, masks, apply_sigmoid=apply_sigmoid).item()

    epoch_time = time.time() - start_time
    return total_loss / len(loader), total_acc / len(loader), epoch_time


def train_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler,
    num_epochs, device, experiment_dir,
    use_scheduler=False, use_early_stopping=False, patience=10,
    apply_sigmoid=True,
):
    training_metrics_csv_path = os.path.join(experiment_dir, "training_metrics.csv")
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
        "train_time": [], "val_time": [],
    }

    best_val_loss = float("inf")
    early_stopping_counter = 0
    training_metrics_info = []

    for epoch in range(num_epochs):
        train_loss, train_acc, train_time = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            apply_sigmoid=apply_sigmoid) 
        val_loss, val_acc, val_time = validate_one_epoch(
            model, val_loader, criterion, device,
            apply_sigmoid=apply_sigmoid) 

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_time"].append(train_time)
        history["val_time"].append(val_time)

        print(
            f"Epoch {epoch+1}/{num_epochs} "
            f"- Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} "
            f"- Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} "
            f"- LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        training_metrics_info.append([
            epoch + 1, train_loss, val_loss,
            train_acc, val_acc, train_time, val_time,
        ])

        if use_scheduler and scheduler is not None:
            scheduler.step(val_loss)

        if use_early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print("Early stopping triggered!")
                break

    df = pd.DataFrame(training_metrics_info, columns=[
        "epoch", "loss_training", "loss_validation",
        "accuracy_training", "accuracy_validation",
        "train_epoch_time", "val_epoch_time",
    ])
    df.to_csv(training_metrics_csv_path, index=False)
    print(f"Training/validation metrics saved at {training_metrics_csv_path}.")
    return history


def test_and_save_results(
    model, test_loader, test_dataset, device,
    experiment_dir, segmentation_mask_dir,
    apply_sigmoid=True,   # <-- False para UNetProtoSeg (já tem softmax)
):
    metrics_csv_path     = os.path.join(experiment_dir, "segmentation_metrics.csv")
    metrics_summary_path = os.path.join(experiment_dir, "metrics_summary.txt")

    with open(metrics_csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image_name", "class", "IoU", "Dice",
                         "Precision", "Recall", "Specificity", "Accuracy"])

    iou_scores, dice_scores       = [], []
    precision_scores, recall_scores = [], []
    specificity_scores, accuracy_scores = [], []
    inference_times = []

    for class_name in test_dataset.get_classes():
        os.makedirs(os.path.join(segmentation_mask_dir, class_name), exist_ok=True)

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            images      = batch["image"].to(device)
            masks       = batch["mask"].to(device)
            labels      = batch["label"]
            image_names = batch["image_name"]

            inference_start = time.time()
            outputs = model(images)

            # ✅ Aplica sigmoid apenas para modelos que retornam logits
            if apply_sigmoid:
                outputs = torch.sigmoid(outputs)

            outputs = (outputs > 0.5).float()
            inference_end = time.time()

            num_images = images.size(0)
            inference_time_per_image_ms = (
                (inference_end - inference_start) / num_images
            ) * 1000.0
            inference_times.append(inference_time_per_image_ms)

            for i in range(num_images):
                original_name    = image_names[i]
                iou_value        = iou(outputs[i], masks[i]).item()
                dice_value       = dice_coeff(outputs[i], masks[i]).item()
                precision_value, recall_value = precision_recall(outputs[i], masks[i])
                precision_value  = precision_value.item()
                recall_value     = recall_value.item()
                specificity_value = specificity(outputs[i], masks[i]).item()
                accuracy_value   = accuracy(outputs[i], masks[i]).item()

                iou_scores.append(iou_value)
                dice_scores.append(dice_value)
                precision_scores.append(precision_value)
                recall_scores.append(recall_value)
                specificity_scores.append(specificity_value)
                accuracy_scores.append(accuracy_value)

                class_index = labels[i].item()
                class_name  = test_dataset.get_classes()[class_index]

                with open(metrics_csv_path, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([original_name, class_name, iou_value,
                                     dice_value, precision_value, recall_value,
                                     specificity_value, accuracy_value])

                img_dir = os.path.join(segmentation_mask_dir, class_name)
                os.makedirs(img_dir, exist_ok=True)

                original_img = transforms.ToPILImage()(images[i].cpu())
                mask_img     = transforms.ToPILImage()(masks[i].cpu())
                pred_img     = Image.fromarray(
                    (outputs[i].cpu().numpy().squeeze() * 255).astype(np.uint8))

                original_img.save(os.path.join(img_dir, f"{original_name}.png"))
                mask_img.save(os.path.join(img_dir, f"{original_name}-mask.png"))
                pred_img.save(os.path.join(img_dir, f"{original_name}-seg.png"))

    with open(os.path.join(experiment_dir, "inference_times.txt"), "w") as f:
        f.write(f"Inference time per image: "
                f"{np.mean(inference_times):.4f} ms ± {np.std(inference_times):.4f} ms\n")

    df = pd.read_csv(metrics_csv_path)
    metrics_by_class = df.groupby("class").agg({
        "IoU":         ["mean", "std"],
        "Dice":        ["mean", "std"],
        "Precision":   ["mean", "std"],
        "Recall":      ["mean", "std"],
        "Specificity": ["mean", "std"],
        "Accuracy":    ["mean", "std"],
    })
    metrics_by_class.columns = [f"{m}_{s}" for m, s in metrics_by_class.columns]
    class_metrics_csv_path = os.path.join(experiment_dir, "segmentation_metrics_class.csv")
    metrics_by_class.to_csv(class_metrics_csv_path)
    print(f"Per-class metrics saved at {class_metrics_csv_path}")

    with open(metrics_summary_path, "w") as f:
        f.write(f"IoU:         {np.mean(iou_scores):.4f} ± {np.std(iou_scores):.4f}\n")
        f.write(f"Dice:        {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}\n")
        f.write(f"Precision:   {np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}\n")
        f.write(f"Recall:      {np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}\n")
        f.write(f"Specificity: {np.mean(specificity_scores):.4f} ± {np.std(specificity_scores):.4f}\n")
        f.write(f"Accuracy:    {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}\n")

    print(f"Metrics summary saved at {metrics_summary_path}")