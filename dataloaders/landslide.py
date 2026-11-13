"""
Few-shot segmentation dataloader for the Landslide4Sense dataset.

Only the RGB bands (B2/B3/B4) of the 14-band multispectral stack are used,
together with the binary landslide label. Output is episodic and matches the
input contract of ``models.fewshot.FewShotSeg.forward``:

    supp_imgs   : list[way] -> list[shot] -> FloatTensor [B, C, H, W]
    supp_mask   : list[way] -> list[shot] -> FloatTensor [B, H, W]   (0/1)
    qry_imgs    : list[query] -> FloatTensor [B, C, H, W]
    qry_mask    : FloatTensor [B, H, W]                              (0/1)

Band mapping (0-based index into the 14-band h5 array):
    B1=0, B2=1, B3=2, B4=3, ... , B12=11, slope=12, DEM=13
So RGB (B2/B3/B4) -> indices (1, 2, 3).

Dataset layout (see datasets/landslide4sense/README.md):
    <data_dir>/images/<split>/image_<i>.h5        # dataset key 'img', shape (128, 128, 14)
    <data_dir>/annotations/<split>/mask_<i>.h5    # dataset key 'mask', shape (128, 128), 0/1
"""

import glob
import os
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    import cv2  # optional; needed only for arbitrary-angle rotation
except Exception:  # pragma: no cover
    cv2 = None

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# (ImageNet stats) applied on the selected RGB bands.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Per-band stats from the original Landslide4Sense loader (dataset.py), subset
# at B2/B3/B4 (indices 1,2,3).
LANDSLIDE_MEAN = (-0.3074, -0.1277, -0.0625)
LANDSLIDE_STD = (0.8775, 0.8860, 0.8869)

NORM_PARAMS = {
    "imagenet": (IMAGENET_MEAN, IMAGENET_STD),
    "landslide": (LANDSLIDE_MEAN, LANDSLIDE_STD),
    "none": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
}

# Landslide4Sense class names (label 0 = background, label 1 = landslide).
LABEL_NAMES = {0: "Non-Landslide", 1: "Landslide"}

# Default location (Colab / Google Drive).
DEFAULT_DATA_DIR = "/content/drive/MyDrive/thesis/datasets/landslide4sense"

# Process-wide caches to cut repeated disk (Drive) access:
#   _MASK_FG_CACHE : mask_path -> bool   (foreground presence)
#   _INDEX_CACHE   : (data_dir, split) -> list[(img_path, mask_path)]
_MASK_FG_CACHE: Dict[str, bool] = {}
_INDEX_CACHE: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}


def get_label_names() -> Dict[int, str]:
    """Return {label_code: class_name} for the dataset."""
    return dict(LABEL_NAMES)


def get_norm_params(normalize: str = "imagenet"):
    """Return (mean, std) tuples for the requested normalization scheme."""
    if normalize not in NORM_PARAMS:
        raise ValueError(
            f"Unknown normalize='{normalize}', expected one of {list(NORM_PARAMS)}"
        )
    return NORM_PARAMS[normalize]


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


class LandslideFewShotDataset(Dataset):
    """Episodic few-shot dataset over a single Landslide4Sense split.

    Each ``__getitem__`` returns one episode (support + query) sampled from the
    same split. ``n_way`` is forced to 1 because Landslide4Sense is a single
    foreground class.
    """

    SPLIT_ALIASES = {
        "train": "train",
        "val": "validation",
        "valid": "validation",
        "validation": "validation",
        "test": "test",
    }

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        rgb_bands: Sequence[int] = (1, 2, 3),  # B2, B3, B4
        n_way: int = 1,
        n_shot: int = 1,
        n_query: int = 1,
        normalize: str = "imagenet",
        input_scale: float = 1.0,
        require_fg: bool = True,
        min_fg_pixels: int = 1,
        max_iters: Optional[int] = None,
        img_size: Optional[int] = None,
        seed: int = 2022,
        # augmentation
        aug_flip: bool = False,
        aug_rot90: bool = False,
        aug_color: bool = False,
        aug_rotation: float = 0.0,
        color_brightness: float = 0.2,
        color_contrast: float = 0.2,
        color_gamma: float = 0.2,
        color_gain: float = 0.1,
        # channels (within the selected bands) that are NOT spectral: color
        # augmentation is skipped for these. Landslide4Sense topography = slope
        # (band idx 12) and DEM (band idx 13); RGB-only loaders leave this empty.
        topo_channels: Sequence[int] = (),
    ):
        if split not in self.SPLIT_ALIASES:
            raise ValueError(
                f"Unknown split='{split}', expected one of {list(self.SPLIT_ALIASES)}"
            )
        if n_way != 1:
            raise ValueError(
                "Landslide4Sense is single-class; n_way must be 1."
            )
        if n_shot < 1 or n_query < 1:
            raise ValueError("n_shot and n_query must be >= 1.")

        self.data_dir = os.path.abspath(data_dir)
        self.split = self.SPLIT_ALIASES[split]
        self.rgb_bands = tuple(rgb_bands)
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.normalize = normalize
        self.input_scale = float(input_scale)
        self.require_fg = require_fg
        self.min_fg_pixels = int(min_fg_pixels)
        self.img_size = img_size
        self.seed = seed

        # augmentation config
        self.aug_flip = aug_flip
        self.aug_rot90 = aug_rot90
        self.aug_color = aug_color
        self.aug_rotation = float(aug_rotation)
        self.color_brightness = float(color_brightness)
        self.color_contrast = float(color_contrast)
        self.color_gamma = float(color_gamma)
        self.color_gain = float(color_gain)
        # Indices (within the loaded bands) that are spectral -> colour aug only.
        topo = set(int(c) for c in topo_channels)
        self.spectral_idx = [i for i in range(len(self.rgb_bands)) if i not in topo]

        mean, std = get_norm_params(normalize)
        self.mean = np.asarray(mean, dtype=np.float32).reshape(len(self.rgb_bands), 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(len(self.rgb_bands), 1, 1)

        # episode counter, used only to vary the sampling RNG per access
        self._ep = 0
        self._rng = np.random.RandomState(seed)

        self.files = self._index_split(self.split)
        if len(self.files) == 0:
            raise FileNotFoundError(
                f"No image/mask pairs found under {self.data_dir} for split "
                f"'{self.split}'."
            )

        self._episodes_per_epoch = len(self.files)
        if max_iters is not None:
            self._episodes_per_epoch = int(max_iters)

        # Cache of episode-count for debugging/statistics.
        self.last_episode_support = None
        self.last_episode_query = None

    # -- indexing -----------------------------------------------------------

    def _index_split(self, split: str) -> List[Tuple[str, str]]:
        key = (self.data_dir, split)
        cached = _INDEX_CACHE.get(key)
        if cached is not None:
            return list(cached)
        img_dir = os.path.join(self.data_dir, "images", split)
        mask_dir = os.path.join(self.data_dir, "annotations", split)
        pairs: List[Tuple[str, str]] = []
        for img_path in sorted(glob.glob(os.path.join(img_dir, "image_*.h5"))):
            stem = os.path.basename(img_path)
            mask_path = os.path.join(mask_dir, stem.replace("image", "mask"))
            if os.path.isfile(mask_path):
                pairs.append((img_path, mask_path))
        _INDEX_CACHE[key] = list(pairs)
        return pairs

    # -- helpers ------------------------------------------------------------

    def _read_h5(self, path: str, key: str) -> np.ndarray:
        with h5py.File(path, "r") as hf:
            arr = hf[key][:]
        return np.asarray(arr)

    def _load_rgb(self, img_path: str) -> np.ndarray:
        image = self._read_h5(img_path, "img").astype(np.float32)
        if image.ndim == 2:  # degenerate: single band stored as 2D
            image = image[:, :, None]
        # image: (H, W, 14) -> select RGB -> (H, W, C) -> (C, H, W)
        image = image[:, :, list(self.rgb_bands)]
        image = np.transpose(image, (2, 0, 1))
        image = np.ascontiguousarray(image, dtype=np.float32)
        if self.img_size is not None:
            image = self._resize(image, self.img_size, mode="bilinear")
        return image

    def _load_mask(self, mask_path: str) -> np.ndarray:
        mask = self._read_h5(mask_path, "mask").astype(np.float32)
        if mask.ndim == 3:  # (H, W, 2) one-hot -> take landslide channel if present
            # Landslide4Sense stores single-channel (H, W); guard for one-hot.
            if mask.shape[-1] == 2:
                mask = mask[..., 1]
            else:
                mask = mask[..., 0]
        mask = (mask > 0.5).astype(np.float32)
        if self.img_size is not None:
            # Nearest-neighbour keeps labels binary; bilinear on image only.
            mask = self._resize(mask[None], self.img_size, mode="nearest")[0]
        return np.ascontiguousarray(mask, dtype=np.float32)

    @staticmethod
    def _resize(arr: np.ndarray, size: int, mode: str = "nearest") -> np.ndarray:
        """Resize (C,H,W) or (1,H,W) to (size,size).

        ``mode='bilinear'`` for multispectral images, ``mode='nearest'`` for
        labels (keeps 0/1 binary).
        """
        t = torch.from_numpy(arr).float().unsqueeze(0)
        kw = {}
        if mode == "bilinear":
            kw["align_corners"] = False
        t = torch.nn.functional.interpolate(
            t, size=(size, size), mode=mode, **kw
        )
        return t.squeeze(0).numpy()

    def _normalize_(self, image: np.ndarray) -> np.ndarray:
        if self.input_scale != 1.0:
            image = image * self.input_scale
        image = (image - self.mean) / self.std
        return np.ascontiguousarray(image, dtype=np.float32)

    # -- augmentation -------------------------------------------------------

    def _geom_aug(
        self, img: np.ndarray, mask: np.ndarray, rng: np.random.RandomState
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Geometric augmentation applied identically to image and mask.

        flip: random horizontal / vertical mirror.
        rot90: random k*90-degree rotation (k in {0,1,2,3}).
        rotation: random angle in [-aug_rotation, aug_rotation] degrees
            (bilinear for image, nearest for mask; needs cv2).
        """
        if self.aug_flip:
            if rng.rand() < 0.5:  # horizontal
                img = img[:, :, ::-1]
                mask = mask[:, ::-1]
            if rng.rand() < 0.5:  # vertical
                img = img[:, ::-1, :]
                mask = mask[::-1, :]
        if self.aug_rot90:
            k = int(rng.randint(0, 4))
            if k:
                img = np.rot90(img, k, axes=(1, 2))
                mask = np.rot90(mask, k, axes=(0, 1))
        if self.aug_rotation > 0 and cv2 is not None:
            img, mask = self._rotate(img, mask, rng)
        return (
            np.ascontiguousarray(img, dtype=np.float32),
            np.ascontiguousarray(mask, dtype=np.float32),
        )

    def _rotate(
        self, img: np.ndarray, mask: np.ndarray, rng: np.random.RandomState
    ) -> Tuple[np.ndarray, np.ndarray]:
        angle = float(rng.uniform(-self.aug_rotation, self.aug_rotation))
        if abs(angle) < 1e-3:
            return img, mask
        C, H, W = img.shape
        center = (W / 2.0, H / 2.0)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img_r = np.empty_like(img)
        for c in range(C):
            img_r[c] = cv2.warpAffine(
                img[c], M, (W, H), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
        mask_r = cv2.warpAffine(
            mask, M, (W, H), flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
        )
        return img_r, (mask_r > 0.5).astype(np.float32)

    def _color_aug(self, image: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        """Photometric perturbation on spectral channels only.

        Topography channels (``topo_channels``) are left untouched, matching the
        Landslide4Sense protocol ("color perturbation is only used for
        multi-spectral data, not DEM and slope data"). Applied before
        normalization.
        """
        image = image.copy()
        idx = self.spectral_idx
        if not idx:
            return image

        band = image[idx, :, :]
        if self.color_brightness > 0:
            band = band + rng.uniform(-self.color_brightness, self.color_brightness)
        if self.color_contrast > 0:
            c = 1.0 + rng.uniform(-self.color_contrast, self.color_contrast)
            band_mean = band.mean(axis=(1, 2), keepdims=True)
            band = (band - band_mean) * c + band_mean
        if self.color_gamma > 0:
            g = 1.0 + rng.uniform(-self.color_gamma, self.color_gamma)
            band = np.clip(band, 1e-6, None) ** g
        if self.color_gain > 0:  # per-channel multiplicative gain
            gain = 1.0 + rng.uniform(-self.color_gain, self.color_gain, size=(len(idx), 1, 1))
            band = band * gain

        image[idx, :, :] = band.astype(np.float32)
        return image

    def _has_fg(self, mask: np.ndarray) -> bool:
        return bool(mask.sum() >= self.min_fg_pixels)

    def _mask_has_fg(self, mask_path: str) -> bool:
        """Foreground test with a process-wide cache.

        Avoids re-reading (potentially over Drive) the same mask on every
        sampling attempt; each mask file is read at most once per process.
        """
        cached = _MASK_FG_CACHE.get(mask_path)
        if cached is not None:
            return cached
        mask = self._read_h5(mask_path, "mask")
        if mask.ndim == 3:
            mask = mask[..., -1] if mask.shape[-1] == 2 else mask[..., 0]
        has = bool((mask > 0).sum() >= self.min_fg_pixels)
        _MASK_FG_CACHE[mask_path] = has
        return has

    def _sample_pair(self, rng: np.random.RandomState, need_fg: bool) -> Tuple[str, str]:
        n = len(self.files)
        if not (self.require_fg and need_fg):
            return self.files[rng.randint(0, n)]
        for _ in range(self._MAX_SAMPLE_TRIES):
            idx = rng.randint(0, n)
            img_path, mask_path = self.files[idx]
            if self._mask_has_fg(mask_path):
                return img_path, mask_path
        # Fallback: relax the constraint rather than loop forever.
        return self.files[rng.randint(0, n)]

    _MAX_SAMPLE_TRIES = 100

    # -- dataset protocol ---------------------------------------------------

    def __len__(self) -> int:
        return self._episodes_per_epoch

    def __getitem__(self, index: int) -> Dict[str, object]:
        # Deterministic-but-varied RNG per item; reseeded by worker_init_fn.
        rng = np.random.RandomState((self.seed + index + self._ep) % (2 ** 31 - 1))
        self._ep += 1

        H = W = self.img_size  # resolved on first load if None
        support_imgs, support_masks, support_names = [], [], []
        for s in range(self.n_shot):
            img_path, mask_path = self._sample_pair(rng, need_fg=True)
            img = self._load_rgb(img_path)
            msk = self._load_mask(mask_path)
            img, msk = self._geom_aug(img, msk, rng)
            support_imgs.append(img)
            support_masks.append(msk)
            support_names.append(os.path.basename(img_path))

        query_imgs, query_masks, query_names = [], [], []
        for q in range(self.n_query):
            img_path, mask_path = self._sample_pair(rng, need_fg=True)
            img = self._load_rgb(img_path)
            msk = self._load_mask(mask_path)
            img, msk = self._geom_aug(img, msk, rng)
            query_imgs.append(img)
            query_masks.append(msk)
            query_names.append(os.path.basename(img_path))

        C, Hs, Ws = support_imgs[0].shape
        support_imgs = [self._color_aug(im, rng) if self.aug_color else im
                        for im in support_imgs]
        query_imgs = [self._color_aug(im, rng) if self.aug_color else im
                      for im in query_imgs]

        support_imgs_t = torch.from_numpy(
            self._normalize_(np.stack(support_imgs, axis=0))
        ).float()
        support_masks_t = torch.from_numpy(np.stack(support_masks, axis=0)).float()
        query_imgs_t = torch.from_numpy(
            self._normalize_(np.stack(query_imgs, axis=0))
        ).float()
        query_masks_t = torch.from_numpy(np.stack(query_masks, axis=0)).float()

        # Reshape to [n_way(=1), n_shot, C, H, W] / [n_way, n_shot, H, W]
        support_imgs_t = support_imgs_t.unsqueeze(0)  # [1, n_shot, C, H, W]
        support_masks_t = support_masks_t.unsqueeze(0)  # [1, n_shot, H, W]

        self.last_episode_support = support_names
        self.last_episode_query = query_names

        return {
            "support_imgs": support_imgs_t,   # [1, n_shot, C, H, W]
            "support_masks": support_masks_t,  # [1, n_shot, H, W]
            "query_imgs": query_imgs_t,      # [n_query, C, H, W]
            "query_masks": query_masks_t,    # [n_query, H, W]
            "support_names": support_names,
            "query_names": query_names,
        }


# --------------------------------------------------------------------------- #
# Collate
# --------------------------------------------------------------------------- #


def episodic_collate(batch: List[Dict[str, object]]) -> Dict[str, object]:
    """Collate episodic samples into the nested structure ``FewShotSeg`` wants.

    With batch size B and n_query == 1:
        support_imgs : list[way] -> list[shot] -> Tensor [B, C, H, W]
        support_masks: list[way] -> list[shot] -> Tensor [B, H, W]
        query_imgs   : list[query] -> Tensor [B, C, H, W]
        query_masks  : Tensor [B, H, W]
    """
    assert len(batch) >= 1
    n_way, n_shot = batch[0]["support_imgs"].shape[:2]
    n_query = batch[0]["query_imgs"].shape[0]

    support_imgs: List[List[List[torch.Tensor]]] = []
    support_masks: List[List[List[torch.Tensor]]] = []
    for w in range(n_way):
        way_imgs, way_masks = [], []
        for s in range(n_shot):
            shot_imgs = torch.stack([b["support_imgs"][w, s] for b in batch], dim=0)
            shot_masks = torch.stack([b["support_masks"][w, s] for b in batch], dim=0)
            way_imgs.append(shot_imgs)   # [B, C, H, W]
            way_masks.append(shot_masks)  # [B, H, W]
        support_imgs.append(way_imgs)
        support_masks.append(way_masks)

    query_imgs: List[torch.Tensor] = []
    for q in range(n_query):
        query_imgs.append(torch.stack([b["query_imgs"][q] for b in batch], dim=0))

    # ``FewShotSeg.forward`` indexes qry_mask[epi] and expects [B, H, W] when
    # n_query == 1. Keep [B, n_query, H, W] for the general case.
    query_masks_all = torch.stack([b["query_masks"] for b in batch], dim=0)  # [B, n_q, H, W]
    query_masks = query_masks_all[:, 0] if n_query == 1 else query_masks_all

    return {
        "supp_imgs": support_imgs,
        "supp_mask": support_masks,
        "qry_imgs": query_imgs,
        "qry_mask": query_masks,
        "support_names": [b["support_names"] for b in batch],
        "query_names": [b["query_names"] for b in batch],
    }


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #


def _worker_init_fn(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    if info is not None:
        seed = (info.seed % (2 ** 31 - 1))
        np.random.seed(seed)
        info.dataset.seed = int(seed)


def get_dataloader(
    data_dir: str = DEFAULT_DATA_DIR,
    split: str = "train",
    n_shot: int = 1,
    n_query: int = 1,
    batch_size: int = 1,
    num_workers: int = 0,
    max_iters: Optional[int] = None,
    normalize: str = "imagenet",
    rgb_bands: Sequence[int] = (1, 2, 3),
    input_scale: float = 1.0,
    require_fg: bool = True,
    min_fg_pixels: int = 1,
    img_size: Optional[int] = None,
    seed: int = 2022,
    pin_memory: bool = True,
    aug_flip: bool = False,
    aug_rot90: bool = False,
    aug_color: bool = False,
    aug_rotation: float = 0.0,
    color_brightness: float = 0.2,
    color_contrast: float = 0.2,
    color_gamma: float = 0.2,
    color_gain: float = 0.1,
    topo_channels: Sequence[int] = (),
) -> DataLoader:
    """Build a ``DataLoader`` that yields episodes for ``FewShotSeg``.

    ``split`` accepts 'train', 'val'/'validation' or 'test'. For the MGPD-Net
    protocol use ``split='train'`` for training and ``split='validation'`` for
    evaluation.
    """
    dataset = LandslideFewShotDataset(
        data_dir=data_dir,
        split=split,
        rgb_bands=rgb_bands,
        n_way=1,
        n_shot=n_shot,
        n_query=n_query,
        normalize=normalize,
        input_scale=input_scale,
        require_fg=require_fg,
        min_fg_pixels=min_fg_pixels,
        max_iters=max_iters,
        img_size=img_size,
        seed=seed,
        aug_flip=aug_flip,
        aug_rot90=aug_rot90,
        aug_color=aug_color,
        aug_rotation=aug_rotation,
        color_brightness=color_brightness,
        color_contrast=color_contrast,
        color_gamma=color_gamma,
        color_gain=color_gain,
        topo_channels=topo_channels,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # episodes are sampled inside __getitem__
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=episodic_collate,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
    )
    return loader


if __name__ == "__main__":
    root = os.path.join(os.path.dirname(__file__), "..", "datasets", "landslide4sense")
    loader = get_dataloader(data_dir=root, split="train", n_shot=1, n_query=1, num_workers=0)
    batch = next(iter(loader))
    print("supp_imgs  ways/shots:", len(batch["supp_imgs"]), len(batch["supp_imgs"][0]),
          "tensor:", tuple(batch["supp_imgs"][0][0].shape))
    print("supp_mask  tensor:", tuple(batch["supp_mask"][0][0].shape))
    print("qry_imgs   queries:", len(batch["qry_imgs"]), "tensor:", tuple(batch["qry_imgs"][0].shape))
    print("qry_mask   tensor:", tuple(batch["qry_mask"].shape))
    print("support_names:", batch["support_names"], "query_names:", batch["query_names"])
