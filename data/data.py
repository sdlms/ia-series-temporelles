# ==============================================================================
# data.py — Dataset PyTorch pour les séries temporelles d'images aériennes
# ==============================================================================
#
# Structure attendue des données sur disque :
#   <root_path>/
#     <dep_id>/
#       orthoimage/
#         <building_id>/
#           <YYYY_...>.tif     # Images aériennes (une par date)
#           emprise.tif        # Masque de l'emprise du bâtiment
#
# Fichiers JSON requis (à la racine du projet) :
#   - all_file_paths.json  : dict { dep_id -> { building_id -> [liste de chemins .tif] } }
#   - date_apparition.json : dict { building_id -> année d'apparition du bâtiment }
# ==============================================================================

import os
import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np
import json
from PIL import Image

# Global configuration for tensor precision
DEFAULT_DTYPE = torch.float16  # or torch.float32 for full precision


# Répartition des départements par split
IDS_PER_SPLIT = {
    "val": ["04", "14", "29", "31", "58", "66", "67", "77"],
    "test": ["12", "15", "22", "26", "36", "61", "64", "68", "69", "71", "73", "75", "76", "83", "84", "85", "92", "93", "94"],
}

IDS_PER_SPLIT["train"] = ["01", "02", "03", "05", "06", "07", "09", "2A", "2B"] + [str(k) for k in range(10, 96) if str(k) not in IDS_PER_SPLIT["val"] and str(k) not in IDS_PER_SPLIT["test"] and k != 20] 
IDS_PER_SPLIT["train+val"] = IDS_PER_SPLIT["train"] + IDS_PER_SPLIT["val"]

MEAN = [105.66, 111.35, 102.18, 106.59]
STD = [52.23, 45.62, 44.30, 39.78]


class BuildingTimeSeriesDataset(Dataset):
    """
    Dataset PyTorch pour la tâche de datation d'apparition de bâtiments.

    Pour chaque bâtiment, charge une série temporelle d'images aériennes
    (une image par date disponible) et fournit :
      - la pile d'images (T, 4, H, W) — 4 canaux : R, G, B, IR
      - le masque d'emprise du bâtiment (H, W)
      - l'indice temporel du premier frame où le bâtiment est apparu

    Args:
        root_path (str): Chemin vers le répertoire racine des données.
        split (str): 'train', 'val' ou 'test'.
    """

    def __init__(self, root_path, split="train", norm=False, augment=False, dropout_on=False, drop_prob=0.5, dataset_ext="jpg"):
        self.root_path = root_path
        self.split = split
        self.dep_ids = IDS_PER_SPLIT[self.split]
        self.norm = norm
        self.augment = augment 
        self.dropout_on = dropout_on
        self.drop_prob = drop_prob
        self.dataset_ext = dataset_ext

        # Année d'apparition de chaque bâtiment (référence pour le label)
        self.date_apparition = json.load(open("data/date_apparition.json", "r"))

        # Charge la liste des échantillons et les métadonnées associées
        self.dep_ids, self.building_ids, self.samples = self._find_samples()

        print(f"Split '{split}' chargé — {len(self.samples)} bâtiments.")

    def _find_samples(self):
        """
        Lit all_file_paths.json et construit les listes parallèles
        (dep_ids, building_ids, samples) pour tous les départements du split.

        Returns:
            dep_ids      (list[str]): Identifiant de département pour chaque sample.
            building_ids (list[str]): Identifiant de bâtiment pour chaque sample.
            samples      (list[list[str]]): Liste de chemins .tif pour chaque bâtiment.
        """
        all_file_paths = json.load(open("data/all_file_paths.json", "r"))
        samples, building_ids, dep_ids = [], [], []

        for dep in self.dep_ids:
            dep_data = all_file_paths[dep]
            samples.extend(dep_data.values())
            building_ids.extend(dep_data.keys())
            dep_ids.extend([dep] * len(dep_data))
        
        # Retirer les batiments dont la date d'apparition est inconnue ou mal définie (null) 
        building_ids_corrected = [b for b in building_ids if self.date_apparition.get(b) is not None]
        dep_ids_corrected = [dep_ids[i] for i, b in enumerate(building_ids) if self.date_apparition.get(b) is not None]
        samples_corrected = [samples[i] for i, b in enumerate(building_ids) if self.date_apparition.get(b) is not None]
        
        return dep_ids_corrected, building_ids_corrected, samples_corrected

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Charge et retourne un échantillon.

        Returns:
            dict avec les clés :
              'images'     : Tensor (T, 4, H, W) — série temporelle d'images
              'emprise'    : Tensor (H, W)        — masque de l'emprise du bâtiment
              'frame_id'   : int                  — indice du premier frame post-apparition
              'n_channels' : Tensor (T, 4)        — masque de validité des canaux par frame
        """
        tif_files = self.samples[idx]
        building_id = self.building_ids[idx]
        dep_id = self.dep_ids[idx]

        year_apparition = int(self.date_apparition[building_id])

        images, years, n_channels = [], [], []
        
        # Pre-compute normalization constants once (more efficient)
        if self.norm:
            mean_arr = np.array(MEAN)[:, None, None]
            std_arr = np.array(STD)[:, None, None]

        for path in tif_files:
            # Extraction de l'année depuis le nom de fichier (format : YYYY_...)
            years.append(int(path[:4]))

            if self.dataset_ext == "tif":
                with rasterio.open(
                    os.path.join(self.root_path, dep_id, "orthoimage", building_id, path)
                ) as src:
                    # Load as float32, then convert to float16 if needed (saves memory during compute)
                    image = src.read().astype(np.float32)  # (C, H, W)
            elif self.dataset_ext in ["jpg", "jpeg", "png"]:
                curr_path = os.path.join(self.root_path, dep_id, "orthoimage", building_id, path.replace('tif', self.dataset_ext))
                image = np.array(Image.open(curr_path))
                if image.ndim == 2:  # triple du canal IR en cas d'image IR seule
                    image = np.stack([image, image, image], axis=2)
                image = image.transpose(2, 0, 1).astype(np.float32)  # (C, H, W)
            else:
                raise ValueError(f"Unsupported dataset_ext='{self.dataset_ext}'. Should be 'tif', 'jpg', 'jpeg', or 'png'.")

            # Harmonisation à 4 canaux (R, G, B, IR)
            if image.shape[0] == 3:
                # Image RGB uniquement : ajout d'un canal IR nul
                image = np.concatenate(
                    [image, np.zeros_like(image[0])[None]], axis=0
                )
                n_channels.append([1, 1, 1, 0])

            elif image.shape[0] == 1:
                # Image IR uniquement : ajout de 3 canaux RGB nuls
                zeros = np.zeros_like(image[0])[None]
                image = np.concatenate([zeros, zeros, zeros, image], axis=0)
                n_channels.append([0, 0, 0, 1])

            else:
                # Image RGBIR complète
                n_channels.append([1, 1, 1, 1])

            if self.dropout_on:
                # Randomly drop channels with probability drop_prob (for data augmentation)
                drop_mask = np.random.rand(4) < self.drop_prob
                image[drop_mask] = 0
            
            # Apply normalization in float32 for numerical stability, then convert
            if self.norm:
                image = (image - mean_arr) / std_arr

            images.append(image)

        # --- Calcul du label : premier frame où le bâtiment est visible ---
        # On cherche le premier indice t tel que years[t] >= year_apparition
        first_frame_id = 0
        for frame_id, year in enumerate(years):
            if year >= year_apparition:
                break
            first_frame_id += 1

        # Chargement du masque d'emprise du bâtiment
        if self.dataset_ext == "tif":
            with rasterio.open(
                os.path.join(
                    self.root_path, dep_id, "orthoimage", building_id, f"emprise.{self.dataset_ext}"
                )
            ) as src:
                emprise = src.read()[0].astype(np.float32)  # (H, W)
        elif self.dataset_ext in ["jpg", "jpeg", "png"]:
            emprise = np.array(Image.open(os.path.join(self.root_path, dep_id, "orthoimage", building_id, f"emprise.png"))).astype(np.float32)[..., 0] / 255  # (H, W)
        else:
            raise ValueError(f"Unsupported dataset_ext='{self.dataset_ext}'. Should be 'tif', 'jpg', 'jpeg', or 'png'.")


        if self.augment:
            images, emprise = d4_transform(images, emprise)

        # Convert to desired dtype after all processing
        images_stacked = np.stack(images, axis=0).astype(np.float16 if DEFAULT_DTYPE == torch.float16 else np.float32)
        emprise = emprise.astype(np.float16 if DEFAULT_DTYPE == torch.float16 else np.float32)

        return {
            "images": torch.from_numpy(images_stacked).to(dtype=DEFAULT_DTYPE),  # (T, 4, H, W)
            "emprise": torch.from_numpy(emprise).to(dtype=DEFAULT_DTYPE),  # (H, W)
            "frame_id": first_frame_id,  # scalaire
            "n_channels": torch.tensor(n_channels, dtype=torch.float16 if DEFAULT_DTYPE == torch.float16 else torch.float32),  # (T, 4)
            "years": torch.tensor(years, dtype=torch.long),  # (T,)
            "building_id": building_id,  # str
        }


def collate_fn_fixed(batch):
    return collate_fn(batch, temp_dim="fixed")


def collate_fn_max(batch):
    return collate_fn(batch, temp_dim="max")
    

def collate_fn(batch, temp_dim="fixed"):
    B = len(batch)
    
    if temp_dim == "fixed":
        max_T = 10
    elif temp_dim == "max":
        max_T = max(item["images"].shape[0] for item in batch)
    else:
        raise ValueError(f"Invalid temp_dim='{temp_dim}'. Should be 'fixed' or 'max'.")
    
    C, H, W = batch[0]["images"].shape[1:]

    # Create tensors with explicit dtype to avoid default float32
    images_padded     = torch.zeros(B, max_T, C, H, W, dtype=DEFAULT_DTYPE)
    n_channels_padded = torch.zeros(B, max_T, 4, dtype=DEFAULT_DTYPE)
    years_padded      = torch.zeros(B, max_T, dtype=torch.long)

    for i, item in enumerate(batch):
        T = item["images"].shape[0]
        images_padded[i, :T]     = item["images"]
        n_channels_padded[i, :T] = item["n_channels"]
        years_padded[i, :T]      = item["years"]

    return {
        "images":      images_padded,
        "emprise":     torch.stack([item["emprise"] for item in batch]).to(dtype=DEFAULT_DTYPE),  # (B, H, W)
        "n_channels":  n_channels_padded,
        "years":       years_padded,
        "frame_id":    torch.tensor([item["frame_id"] for item in batch], dtype=torch.long),
        "building_id": [item["building_id"] for item in batch],
    }


def d4_transform(images, emprise):
    """
    Applies a random D4 dihedral group transformation (rotation/reflection)
    consistently across all T frames and the emprise mask.
    
    Args:
        images  : list of np.ndarray (C, H, W)
        emprise : np.ndarray (H, W)
    Returns:
        images  : list of np.ndarray (C, H, W)
        emprise : np.ndarray (H, W)
    """
    k = np.random.randint(0, 4)   # rotation: 0, 90, 180, 270
    flip = np.random.random() < 0.5

    def apply(x, spatial_axes):
        x = np.rot90(x, k=k, axes=spatial_axes)
        if flip:
            x = np.flip(x, axis=spatial_axes[0])
        return x.copy()

    images = [apply(img, (1, 2)) for img in images]  # spatial axes for (C, H, W)
    emprise = apply(emprise, (0, 1))                  # spatial axes for (H, W)

    return images, emprise