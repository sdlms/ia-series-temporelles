from collections import defaultdict
import csv
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml
from data.data import BuildingTimeSeriesDataset, collate_fn_fixed, collate_fn_max
from models.anysat import AnySatWrapper
from models.flairhub_model import FlairHubWrapper
from models.flairinc_model import FlairIncWrapper
from models.ltae import Ltae
from utils.utils import visualize_time_series, compute_accuracy, compute_mae
from utils.metrics import AverageMeter
from utils.logger import Logger


def train_iteration(config, batch, model, classifier, optimizer, criterion, device='cuda'):
    images    = batch['images'].to(device)    # (B, T, 4, H, W)
    emprise   = batch['emprise'].to(device)   # (B, H, W)
    frame_id  = batch['frame_id'].to(device)  # (B,) — label : indice temporel d'apparition
    n_channels = batch["n_channels"].to(device)  # (B, T, 4)
    years = batch["years"].to(device)  # (B, T)
    building_id = batch["building_id"]  # (B,)

    B, T, C, H, W = images.shape
    images_flat = images.reshape(B * T, C, H, W).float()  # (B*T, C, H, W)
    
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        output = model(images_flat, n_channels)  # (B*T, C_feat, H_out, W_out)
        C_feat, H_out, W_out = output.shape[1], output.shape[2], output.shape[3]
        out = output.view(B, T, C_feat, H_out, W_out)
        emprise_resized = torch.nn.functional.interpolate(
            emprise.unsqueeze(1).to(out.dtype),
            size=(H_out, W_out),
            mode="nearest").squeeze(1)  # (B, H_out, W_out)
        denom = emprise_resized.sum(dim=(1, 2)).clamp(min=1.0)  # (B,)
        weight = emprise_resized / denom[:, None, None]  # (B, H_out, W_out), sums to 1 per b
        pooled = torch.einsum('btchw,bhw->btc', out, weight)  # (B, T, C)
        valid_mask = years > 0  # (B, T)
        pooled = pooled * valid_mask.unsqueeze(-1)  # (B, T, C)
        if config["model"]["classifier"] == "linear":
            pooled = pooled.reshape(B, -1)  
        pooled = pooled.float()

    logits = classifier(pooled)   # (B, N_CLASSES)
    loss = criterion(logits, frame_id)
    loss.backward()
    optimizer.step()
    return logits, loss, frame_id, years


def eval_iteration(config, batch, model, classifier, criterion, device='cuda'):
    images    = batch['images'].to(device)    # (B, T, 4, H, W)
    emprise   = batch['emprise'].to(device)   # (B, H, W)
    frame_id  = batch['frame_id'].to(device)  # (B,) — label : indice temporel d'apparition
    n_channels = batch["n_channels"].to(device)  # (B, T, 4)
    years = batch["years"].to(device)  # (B, T)
    building_id = batch["building_id"]  # (B,)

    B, T, C, H, W = images.shape
    images_flat = images.reshape(B * T, C, H, W).float()  # (B*T, C, H, W)
    
    with torch.no_grad():
        output = model(images_flat, n_channels)  # (B*T, C_feat, H_out, W_out)
        C_feat, H_out, W_out = output.shape[1], output.shape[2], output.shape[3]
        out = output.view(B, T, C_feat, H_out, W_out)
        emprise_resized = torch.nn.functional.interpolate(
            emprise.unsqueeze(1).to(out.dtype),
            size=(H_out, W_out),
            mode="nearest").squeeze(1)  # (B, H_out, W_out)
        denom = emprise_resized.sum(dim=(1, 2)).clamp(min=1.0)  # (B,)
        weight = emprise_resized / denom[:, None, None]  # (B, H_out, W_out), sums to 1 per b
        pooled = torch.einsum('btchw,bhw->btc', out, weight)  # (B, T, C)
        valid_mask = years > 0  # (B, T)
        pooled = pooled * valid_mask.unsqueeze(-1)  # (B, T, C)
        if config["model"]["classifier"] == "linear":
            pooled = pooled.reshape(B, -1)  
        pooled = pooled.float()
        logits = classifier(pooled)   # (B, 10)
        loss = criterion(logits, frame_id)
    return logits, loss, frame_id, years


if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    EXP_NAME = config["exp_name"]
    save_path = os.path.join("results", EXP_NAME)
    os.makedirs(save_path, exist_ok=True)

    device = config["env"]["device"]
    NUM_WORKERS = config["env"]["num_workers"]

    BACKBONE = config["model"]["backbone"]
    RGB_ONLY = config["model"]["rgb_only"]
    MODEL_SIZE = config["model"]["size"]
    USE_FLOAT16 = config["model"]["use_float16"]
    FUSE_MODE = config["model"]["fuse_mode"]
    CLASSIFIER = config["model"]["classifier"]

    ROOT_PATH = config["data"]["root_path"]
    DATASET_EXT = config["data"]["dataset_ext"]
    BATCH_SIZE = config["data"]["batch_size"]
    N_YEARS = config["data"].get("n_years", 26)
    N_CLASSES = config["data"]["n_classes"]
    collate_fn = collate_fn_fixed if config["data"]["collate_fn"] == "fixed" else collate_fn_max

    LEARNING_RATE = config["training"]["learning_rate"]
    WEIGHT_DECAY = config["training"]["weight_decay"]
    NUM_EPOCHS = config["training"]["num_epochs"]
    PRINT_INTERVAL = config["training"]["print_interval"]
    RESUME = config["training"]["resume"]

    # Save config in results folder for reproducibility
    with open(os.path.join(save_path, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    # ------------------------------------------------------------------------------
    # 1. Chargement des datasets
    # ------------------------------------------------------------------------------
    train_dataset = BuildingTimeSeriesDataset(root_path=ROOT_PATH, 
                                              split="train",
                                              norm=True,
                                              augment=config["data"]["data_augmentation"], 
                                              dropout_on=config["data"]["channel_dropout"],
                                              dataset_ext=DATASET_EXT
                                              )
    val_dataset = BuildingTimeSeriesDataset(root_path=ROOT_PATH,
                                            split="val",
                                            norm=True,
                                            augment=False,
                                            dataset_ext=DATASET_EXT
                                            )
    test_dataset = BuildingTimeSeriesDataset(root_path=ROOT_PATH,
                                             split="test",
                                             norm=True,
                                             augment=False,
                                             dataset_ext=DATASET_EXT
                                             )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, drop_last=False, collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, drop_last=False, collate_fn=collate_fn
    )

    # ------------------------------------------------------------------------------
    # 2. Chargement du modèle
    # ------------------------------------------------------------------------------

    if BACKBONE == "flairhub":
        feature_dim = 1920 if RGB_ONLY or FUSE_MODE == "average" else 3840
        model = FlairHubWrapper(
            rgb_only=RGB_ONLY,
            fuse_mode=FUSE_MODE,
            use_float16=USE_FLOAT16
        )
    elif BACKBONE == "flairinc":
        feature_dim = 960 if RGB_ONLY or FUSE_MODE == "average" else 1920
        model = FlairIncWrapper(
            rgb_only=RGB_ONLY,
            fuse_mode=FUSE_MODE,
            use_float16=USE_FLOAT16
        )  
    elif BACKBONE == "anysat":
        feature_dim = 1536 if RGB_ONLY or FUSE_MODE == "average" else 3072
        model = AnySatWrapper(use_float16=USE_FLOAT16)
    else:
        raise ValueError(f"Unknown model {BACKBONE}")

    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    if CLASSIFIER == "linear":
        classifier = torch.nn.Linear(feature_dim * 10, N_CLASSES)
    elif CLASSIFIER == "ltae":
        classifier = Ltae(in_channels=feature_dim, n_classes=N_CLASSES, d_model=None, use_float16=False)
    else:
        raise ValueError(f"Unknown classifier {CLASSIFIER}")

    # ------------------------------------------------------------------------------
    # 3. Boucle d'entraînement
    # ------------------------------------------------------------------------------

    criterion  = torch.nn.functional.cross_entropy
    optimizer  = torch.optim.Adam(classifier.parameters(),
                                  lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    logger = Logger(save_path)
    n_epochs = NUM_EPOCHS
    model = model.to(device)
    classifier = classifier.to(device)
    best_val_acc = 0.
    curr_epoch = 1

    if RESUME and os.path.exists(os.path.join(save_path, 'classifier.pt')):
        ckpt = torch.load(os.path.join(save_path, 'classifier.pt'), map_location=device)
        classifier.load_state_dict(ckpt["classifier"])
        curr_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt["val_acc"]
        logger.restart_from(curr_epoch)
        print(logger.logs)

    print(f"Starting training for {n_epochs} epochs...")
    for epoch in range(curr_epoch, n_epochs + 1):
        print(f"Epoch {epoch}/{n_epochs}")
        # -- Phase entraînement --
        classifier.train()
        train_meter = AverageMeter(n_classes=N_CLASSES, n_years=N_YEARS, device=device)
        for batch_id, batch in enumerate(train_loader):
            logits, loss, frame_id, years = train_iteration(config, batch, model, classifier, optimizer, criterion, device)

            with torch.no_grad():
                pred = torch.argmax(logits, dim=1)          # (B,)
                train_meter.update(loss, pred, frame_id, years)

            if (batch_id + 1) % PRINT_INTERVAL == 0:
                loss, _, _, acc, _, _ = train_meter.get_metrics()
                print(f"      [Iter {batch_id + 1}/{len(train_loader)}] Train loss: {loss:.4f} | Acc: {acc * 100:.2f}%")

        loss, mae, signed_mae, acc, acc1, acc2 = train_meter.get_metrics()
        logger.log({
            "epoch": epoch,
            "train_loss": loss,
            "train_acc": acc,
            "train_acc1": acc1,
            "train_acc2": acc2,
            "train_mae": mae,
            "train_signed_mae": signed_mae
        })
        print(f"   [Epoch {epoch}] Train loss: {loss:.4f} | Acc: {acc * 100:.2f}%")

        # -- Phase de validation --
        classifier.eval()
        val_meter = AverageMeter(n_classes=N_CLASSES, n_years=N_YEARS, device=device)
        with torch.no_grad():
            for batch_id, batch in enumerate(val_loader):
                logits, loss, frame_id,years = eval_iteration(config, batch, model, classifier, criterion, device)

                pred = torch.argmax(logits, dim=1)
                val_meter.update(loss, pred, frame_id, years)

                if (batch_id + 1) % PRINT_INTERVAL == 0:
                    loss, _, _, acc, _, _ = val_meter.get_metrics()
                    print(f"      [Iter {batch_id + 1}/{len(val_loader)}] Val   loss: {loss:.4f} | Acc: {acc * 100:.2f}%")    
        
        loss, mae, signed_mae, acc, acc1, acc2 = val_meter.get_metrics()
        logger.log({
            "val_loss": loss,
            "val_acc": acc,
            "val_acc1": acc1,
            "val_acc2": acc2,
            "val_mae": mae,
            "val_signed_mae": signed_mae
        })
        print(f"   [Epoch {epoch}] Val   loss: {loss:.4f} | Acc: {acc * 100:.2f} | Acc@1: {acc1 * 100:.2f} | Acc@2: {acc2 * 100:.2f} | MAE: {mae:.3f} frames")

        if acc * 100 > best_val_acc:
            best_val_acc = acc * 100
            checkpoint = {
                'epoch': epoch,
                'classifier': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': acc * 100,
                'val_loss': loss
            }
            torch.save(checkpoint, os.path.join(save_path, 'classifier.pt'))

    print(f"Evaluating best checkpoint on test set...")
    checkpoint = torch.load(os.path.join(save_path, 'classifier.pt'))
    classifier.load_state_dict(checkpoint['classifier'])
    classifier.eval()
    test_meter = AverageMeter(n_classes=N_CLASSES, n_years=26, device=device)
    with torch.no_grad():
        for batch_id, batch in enumerate(test_loader):
            logits, loss, frame_id, years = eval_iteration(config, batch, model, classifier, criterion, device)

            pred = torch.argmax(logits, dim=1)
            test_meter.update(loss, pred, frame_id, years)

            if (batch_id + 1) % PRINT_INTERVAL == 0:
                loss, _, _, acc, _, _ = test_meter.get_metrics()
                print(f"  [Iter {batch_id + 1}/{len(test_loader)}] Test loss: {loss:.4f} | Acc: {acc * 100:.2f}%")
    
    loss, mae, signed_mae, acc, acc1, acc2 = test_meter.get_metrics()
    logger.log({
        "test_loss": loss,
        "test_acc": acc,
        "test_acc1": acc1,
        "test_acc2": acc2,
        "test_mae": mae,
        "test_signed_mae": signed_mae
    })
    print(f"Test loss: {loss:.4f} | Acc: {acc * 100:.2f} | Acc@1: {acc1 * 100:.2f} | Acc@2: {acc2 * 100:.2f} | MAE: {mae:.3f} frames | Signed MAE: {signed_mae:.3f} frames")

    confusion_matrix_frame_id = test_meter.conf_mat_frame_id.cpu().numpy()
    confusion_matrix_years = test_meter.conf_mat_year.cpu().numpy()

    df_frame_id = pd.DataFrame(
        confusion_matrix_frame_id,
        index=[f"gt_{i}" for i in range(N_CLASSES)],
        columns=[f"pred_{i}" for i in range(N_CLASSES)],
    )
    df_frame_id.index.name   = "pred_frame_id"
    df_frame_id.columns.name = "gt_frame_id"
    df_frame_id.to_csv(os.path.join(save_path, "confusion_matrix_frame_id.csv"))

    df_years = pd.DataFrame(
        confusion_matrix_years,
        index=[f"gt_{i}" for i in range(2000, 2000 + N_YEARS)],
        columns=[f"pred_{i}" for i in range(2000, 2000 + N_YEARS)],
    )
    df_years.index.name   = "pred_year"
    df_years.columns.name = "gt_year"
    df_years.to_csv(os.path.join(save_path, "confusion_matrix_years.csv"))  