# ==============================================================================
# utils.py — Fonctions utilitaires : visualisation, métriques, helpers
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch


# ------------------------------------------------------------------------------
# Visualisation
# ------------------------------------------------------------------------------

def visualize_time_series(dataset, idx, save_path=None):
    """
    Affiche la série temporelle d'images aériennes d'un bâtiment.
    Le frame d'apparition du bâtiment est mis en évidence en rouge.

    Args:
        dataset   : Instance de BuildingTimeSeriesDataset.
        idx (int) : Indice du sample à visualiser.
        save_path (str, optional): Si fourni, sauvegarde la figure à ce chemin.
    """
    sample       = dataset[idx]
    series       = sample['images']    # (T, 4, H, W)
    highlight_t  = sample['frame_id']
    n_channels   = sample['n_channels']  # (T, 4)
    T            = series.shape[0]

    fig, axes = plt.subplots(1, T, figsize=(4 * T, 4))
    if T == 1:
        axes = [axes]

    for t, ax in enumerate(axes):
        image = series[t].numpy()  # (4, H, W)

        # Sélection des canaux disponibles pour l'affichage RGB
        has_rgb = n_channels[t][:3].sum() == 3
        if has_rgb:
            rgb = image[:3].transpose(1, 2, 0)  # (H, W, 3)
        else:
            # Cas image IR seule : affichage en niveaux de gris
            rgb = np.stack([image[3]] * 3, axis=-1)  # (H, W, 3) en gris

        # Normalisation min-max pour l'affichage
        rgb = _normalize_for_display(rgb)
        ax.imshow(rgb)

        # Titre coloré selon qu'il s'agit ou non du frame d'apparition
        color = 'red' if t == highlight_t else 'black'
        label = f"t={t}\n(apparition)" if t == highlight_t else f"t={t}"
        ax.set_title(label, color=color, fontsize=9)
        ax.axis("off")

    building_id = dataset.building_ids[idx]
    fig.suptitle(f"Bâtiment {building_id} — frame d'apparition : t={highlight_t}", fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure sauvegardée : {save_path}")
    else:
        plt.show()

    plt.close(fig)


def visualize_prediction(dataset, idx, pred_frame_id, save_path=None):
    """
    Affiche la série temporelle avec, en surbrillance :
      - en rouge  : le frame d'apparition réel (ground truth)
      - en vert   : le frame prédit par le modèle

    Args:
        dataset           : Instance de BuildingTimeSeriesDataset.
        idx (int)         : Indice du sample.
        pred_frame_id (int): Frame prédit par le modèle.
        save_path (str, optional): Si fourni, sauvegarde la figure.
    """
    sample      = dataset[idx]
    series      = sample['images']
    gt_frame    = sample['frame_id']
    T           = series.shape[0]

    fig, axes = plt.subplots(1, T, figsize=(4 * T, 4))
    if T == 1:
        axes = [axes]

    for t, ax in enumerate(axes):
        image = series[t].numpy()
        rgb   = image[:3].transpose(1, 2, 0)
        rgb   = _normalize_for_display(rgb)
        ax.imshow(rgb)
        ax.axis("off")

        if t == gt_frame and t == pred_frame_id:
            ax.set_title(f"t={t}\nGT & pred", color='purple', fontsize=9)
        elif t == gt_frame:
            ax.set_title(f"t={t}\nGT", color='red', fontsize=9)
        elif t == pred_frame_id:
            ax.set_title(f"t={t}\npred", color='green', fontsize=9)
        else:
            ax.set_title(f"t={t}", fontsize=9)

    legend = [
        mpatches.Patch(color='red',    label='Ground truth'),
        mpatches.Patch(color='green',  label='Prédiction'),
        mpatches.Patch(color='purple', label='GT = Prédiction'),
    ]
    fig.legend(handles=legend, loc='lower center', ncol=3, fontsize=9)
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close(fig)


def visualize_emprise(dataset, idx, save_path=None):
    """
    Affiche le masque d'emprise (footprint) du bâtiment pour un sample donné.

    Args:
        dataset   : Instance de BuildingTimeSeriesDataset.
        idx (int) : Indice du sample.
        save_path (str, optional): Si fourni, sauvegarde la figure.
    """
    sample  = dataset[idx]
    emprise = sample['emprise'].numpy()  # (H, W)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(emprise, cmap='gray')
    ax.set_title(f"Emprise — bâtiment {dataset.building_ids[idx]}")
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close(fig)


# ------------------------------------------------------------------------------
# Métriques
# ------------------------------------------------------------------------------

def compute_accuracy(preds, targets):
    """
    Calcule la précision exacte (exact match) entre prédictions et labels.

    Args:
        preds   (Tensor ou array, shape B): Frames prédits.
        targets (Tensor ou array, shape B): Frames réels.

    Returns:
        float: Pourcentage de prédictions correctes (0–100).
    """
    if isinstance(preds, torch.Tensor):
        preds   = preds.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    return float((preds == targets).mean() * 100)


def compute_mae(preds, targets):
    """
    Calcule l'erreur absolue moyenne (MAE) en nombre de frames.

    Args:
        preds   (Tensor ou array, shape B): Frames prédits.
        targets (Tensor ou array, shape B): Frames réels.

    Returns:
        float: MAE en frames.
    """
    if isinstance(preds, torch.Tensor):
        preds   = preds.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    return float(np.abs(preds - targets).mean())


# ------------------------------------------------------------------------------
# Helpers internes
# ------------------------------------------------------------------------------

def _normalize_for_display(image):
    """
    Normalise un tableau numpy (H, W, C) entre 0 et 1 pour l'affichage.
    Évite la division par zéro si l'image est uniforme.
    """
    vmin, vmax = image.min(), image.max()
    if vmax - vmin > 1e-6:
        return (image - vmin) / (vmax - vmin)
    return np.zeros_like(image)

def soft_targets(frame_id, valid_mask, sigma=0.5):
    """Cible souple gaussienne centrée sur frame_id, lissée dans l'espace des frames."""
    B, T = valid_mask.shape
    pos = torch.arange(T, device=frame_id.device)[None, :]
    d = (pos - frame_id[:, None]).float()                    # (B, T), signé

    if sigma <= 0:
        w = (d == 0).float()                                 # one-hot
    else:
        w = torch.exp(-d.pow(2) / (2 * sigma ** 2))

    w = w * valid_mask
    return w / w.sum(dim=1, keepdim=True)
