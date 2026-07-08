import json
import os
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from data.data import BuildingTimeSeriesDataset, collate_fn_fixed, collate_fn_max
from models.flairhub_model import FlairHubWrapper
from models.flairinc_model import FlairIncWrapper
from utils.metrics import AverageMeter


def segment_batch(batch, model, device):
    """Renvoie les prédictions (B, T, H, W), l'emprise (B, H, W), frame_id (B,)
    et years (B, T)."""
    images = batch["images"].to(device)            # (B, T, 4, H, W)
    emprise = batch["emprise"].to(device).long()          # (B, H, W)
    frame_id = batch["frame_id"].to(device)        # (B,)
    n_channels = batch["n_channels"].to(device)    # (B, T, 4)
    years = batch["years"].to(device)              # (B, T)
    building_id = batch["building_id"]                      # (B,)

    B, T, C, H, W = images.shape
    images_flat = images.float().reshape(B * T, C, H, W)

    outputs = model(images_flat, n_channels)           # (B*T, K, H, W)
    
    preds = torch.argmax(outputs, dim=1)           # (B*T, H, W)
    preds = preds.reshape(B, T, H, W)
    return preds, emprise, frame_id, years, building_id


def predict_frame_id(preds, emprise, years, seuils, building_class_idx, detection_mode="first"):
    """Pour chaque seuil, renvoie l'indice de la première frame valide où la
    proportion de pixels "bâtiment" dans l'emprise dépasse le seuil.

    Returns: (B, S) tensor of predicted frame indices.
    """
    # preds: (B, T, H, W), emprise: (B, H, W), years: (B, T).
    # On décale les classes prédites de +1 puis on multiplie par l'emprise pour
    # ne conserver que les prédictions à l'intérieur de l'emprise (les pixels
    # hors emprise prennent la valeur 0, distincte de toute classe décalée).
    preds_emprise = (preds + 1) * emprise.unsqueeze(1).long()           # (B, T, H, W)
    masque_batiment = preds_emprise == building_class_idx
    nb_pixels_batiment = masque_batiment.sum(dim=(-2, -1)).float()      # (B, T)
    taille_emprise = emprise.sum(dim=(-2, -1)).float().unsqueeze(1).clamp(min=1.0)  # (B, 1)
    proportions = nb_pixels_batiment / taille_emprise * 100.0           # (B, T)
    seuils_t = torch.tensor(seuils, dtype=torch.float32, device=proportions.device)  # (S,)
    above = proportions.unsqueeze(-1) >= seuils_t.view(1, 1, -1)        # (B, T, S)
    # On masque les frames de padding (years==0 dans le collate par défaut).
    valid_mask = years > 0                                              # (B, T)
    above = above & valid_mask.unsqueeze(-1)
    if detection_mode == "first":  # returns the first frame where the threshold is exceeded
        detection_found = above.any(dim=1)                                  # (B, S)
        first_detection = above.float().argmax(dim=1)
    elif detection_mode == "last":  # returns the last frame where the value is below the threshold + 1
        above_for_cumprod = above | ~valid_mask.unsqueeze(-1)           # (B, T, S)
        above_stable = above_for_cumprod.float().flip(dims=[1]).cumprod(dim=1).flip(dims=[1]).bool() # (B, T, S)
        above_stable = above_stable & valid_mask.unsqueeze(-1)          # (B, T, S)
        detection_found = above_stable.any(dim=1)                       # (B, S)
        first_detection = above_stable.float().argmax(dim=1)            # (B, S)
    last_valid_frame = valid_mask.long().sum(dim=1) - 1                 # (B,)

    pred_frame_id = torch.where(
        detection_found,
        first_detection,
        last_valid_frame.unsqueeze(-1).expand(-1, len(seuils)),
    )                                                                   # (B, S)
    return pred_frame_id


if __name__ == "__main__":
    with open("configs/config_seuil.yaml", "r") as f:
        config = yaml.safe_load(f)

    EXP_NAME = config["exp_name"]
    save_path = os.path.join("results", EXP_NAME)
    os.makedirs(save_path, exist_ok=True)

    device = config["env"]["device"]
    NUM_WORKERS = config["env"]["num_workers"]

    ROOT_PATH = config["data"]["root_path"]
    DATASET_EXT = config["data"]["dataset_ext"]
    BATCH_SIZE = config["data"]["batch_size"]
    N_CLASSES = config["data"]["n_classes"]
    N_YEARS = config["data"].get("n_years", 26)
    BUILDING_CLASS_IDX = config["data"]["building_class_idx"]
    collate_fn = collate_fn_fixed if config["data"]["collate_fn"] == "fixed" else collate_fn_max

    BACKBONE = config["model"]["backbone"]
    USE_FLOAT16 = config["model"]["use_float16"]
    RGB_ONLY = config["model"]["rgb_only"]
    MODEL_SIZE = config["model"]["size"]

    SEUILS = config["method"]["seuils"]
    DETECTION_MODE = config["method"]["detection_mode"]
    PRINT_INTERVAL = config["method"]["print_interval"]

    # Save config in results folder for reproducibility
    with open(os.path.join(save_path, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    # ------------------------------------------------------------------------------
    # 1. Dataset / DataLoader
    # ------------------------------------------------------------------------------
    train_dataset = BuildingTimeSeriesDataset(root_path=ROOT_PATH,
                                            split="train+val",
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
        train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, drop_last=False, collate_fn=collate_fn
    )

    # ------------------------------------------------------------------------------
    # 2. Modèle(s) de segmentation gelé(s)
    # ------------------------------------------------------------------------------
    if BACKBONE == "flairhub":
        model = FlairHubWrapper(
            rgb_only=RGB_ONLY,
            encoder_only=False,
            fuse_mode="average",
            use_float16=USE_FLOAT16,
            size=MODEL_SIZE
        )
    elif BACKBONE == "flairinc":
        model = FlairIncWrapper(
            rgb_only=RGB_ONLY,
            encoder_only=False,
            fuse_mode="average",
            use_float16=USE_FLOAT16,
            size=MODEL_SIZE
        )  
    else:
        raise ValueError(f"Unknown model {BACKBONE}")

    for p in model.parameters():
        p.requires_grad = False
    model.eval().to(device)

    # ------------------------------------------------------------------------------
    # 3. Boucle d'évaluation : un AverageMeter par seuil
    # ------------------------------------------------------------------------------
    meters = [
        AverageMeter(n_classes=N_CLASSES, n_years=N_YEARS, device=device)
        for _ in SEUILS
    ]
    # AverageMeter attend une loss : on passe 0 (pas d'entraînement ici).
    zero_loss = torch.tensor(0.0, device=device)

    print(f"Recherche du seuil sur train+val — {len(train_dataset)} bâtiments, "
          f"{len(SEUILS)} seuils, modèle={BACKBONE}, rgb_only={RGB_ONLY}, size={MODEL_SIZE}.")

    with torch.no_grad():
        for batch_id, batch in enumerate(train_loader):
            preds, emprise, frame_id, years, _ = segment_batch(
                batch, model, device
            )
            pred_frame_id = predict_frame_id(
                preds, emprise, years, SEUILS, BUILDING_CLASS_IDX, DETECTION_MODE
            )  # (B, S)

            for s_idx in range(len(SEUILS)):
                meters[s_idx].update(zero_loss, pred_frame_id[:, s_idx], frame_id, years)

            if (batch_id + 1) % PRINT_INTERVAL == 0:
                print(f"  [Iter {batch_id + 1}/{len(train_loader)}]")

    # ------------------------------------------------------------------------------
    # 4. Résultats : log + matrices de confusion par seuil
    # ------------------------------------------------------------------------------
    summary = {"seuils": list(SEUILS)}
    accs, acc1s, acc2s, maes, signed_maes = [], [], [], [], []
    for s_idx, seuil in enumerate(SEUILS):
        _, mae, signed_mae, acc, acc1, acc2 = meters[s_idx].get_metrics()
        accs.append(acc)
        acc1s.append(acc1)
        acc2s.append(acc2)
        maes.append(mae)
        signed_maes.append(signed_mae)
        print(
            f"[Seuil {seuil}%] "
            f"Acc: {acc * 100:.2f}% | Acc@1: {acc1 * 100:.2f}% | Acc@2: {acc2 * 100:.2f}% "
            f"| MAE: {mae:.3f} frames | Signed MAE: {signed_mae:.3f} frames"
        )

    summary.update({
        "train_acc": accs,
        "train_acc1": acc1s,
        "train_acc2": acc2s,
        "train_mae": maes,
        "train_signed_mae": signed_maes,
    })

    best_seuil_idx = accs.index(max(accs))
    best_seuil = SEUILS[best_seuil_idx]
    print(f"Meilleur seuil sur train+val : {best_seuil}% (Acc={accs[best_seuil_idx] * 100:.2f}%)")

    # Evaluer seuil sur test set
    print(f"Évaluation du seuil sur test — {len(test_dataset)} bâtiments, "
          f"seuil={best_seuil}%, modèle={BACKBONE}, rgb_only={RGB_ONLY}, size={MODEL_SIZE}.")
    test_meter = AverageMeter(n_classes=N_CLASSES, n_years=N_YEARS, device=device)

    all_preds, all_frame_id, all_building_id = [], [], []
    with torch.no_grad():
        for batch_id, batch in enumerate(test_loader):
            preds, emprise, frame_id, years, building_id = segment_batch(
                batch, model, device
            )
            pred_frame_id = predict_frame_id(
                preds, emprise, years, [best_seuil], BUILDING_CLASS_IDX, DETECTION_MODE
            )[:, 0]  # (B,)

            all_preds.extend(pred_frame_id.cpu().numpy().tolist())
            all_frame_id.extend(frame_id.cpu().numpy().tolist())
            all_building_id.extend(building_id)
            test_meter.update(zero_loss, pred_frame_id, frame_id, years)

            if (batch_id + 1) % PRINT_INTERVAL == 0:
                print(f"  [Iter {batch_id + 1}/{len(test_loader)}]")

    _, mae, signed_mae, acc, acc1, acc2 = test_meter.get_metrics()
    print(
        f"[Test - Seuil {best_seuil}%] "
        f"Acc: {acc * 100:.2f}% | Acc@1: {acc1 * 100:.2f}% | Acc@2: {acc2 * 100:.2f}% "
        f"| MAE: {mae:.3f} frames | Signed MAE: {signed_mae:.3f} frames"
    )
    summary.update({
        "best_seuil": best_seuil,
        "test_acc": acc,
        "test_acc1": acc1,
        "test_acc2": acc2,
        "test_mae": mae,
        "test_signed_mae": signed_mae,
    })  
    with open(os.path.join(save_path, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
    
    cm_frame = test_meter.conf_mat_frame_id.cpu().numpy()
    cm_year = test_meter.conf_mat_year.cpu().numpy()

    df_frame = pd.DataFrame(
        cm_frame,
        index=[f"gt_{i}" for i in range(N_CLASSES)],
        columns=[f"pred_{i}" for i in range(N_CLASSES)],
    )
    df_frame.index.name = "gt_frame_id"
    df_frame.columns.name = "pred_frame_id"
    df_frame.to_csv(
        os.path.join(save_path, f"confusion_matrix_frame_id.csv")
    )

    df_year = pd.DataFrame(
        cm_year,
        index=[f"gt_{i}" for i in range(2000, 2000 + N_YEARS)],
        columns=[f"pred_{i}" for i in range(2000, 2000 + N_YEARS)],
    )
    df_year.index.name = "gt_year"
    df_year.columns.name = "pred_year"
    df_year.to_csv(
        os.path.join(save_path, f"confusion_matrix_years.csv")
    )

    output_dict = {b: (gt, pd) for b, gt, pd in zip(all_building_id, all_frame_id, all_preds)}
    with open(os.path.join(save_path, "predictions.json"), "w") as f:
        json.dump(output_dict, f, indent=4)
