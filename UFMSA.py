from __future__ import annotations

import argparse
import contextlib
import csv
import json
import logging
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
try:
    from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
except ImportError:
    AutoImageProcessor = None
    AutoModel = None
    AutoTokenizer = None
try:
    from ray import tune
    from ray.tune.search.basic_variant import BasicVariantGenerator
except ImportError:
    tune = None
    BasicVariantGenerator = None

LOGGER = logging.getLogger("ufmsa")
PAPER_SEEDS: Tuple[int, ...] = (13, 21, 43, 2026, 3407)
LEARNING_RATE_FIELDS: Tuple[str, ...] = (
    "text_learning_rate",
    "visual_learning_rate",
    "frequency_learning_rate",
    "smca_learning_rate",
    "moe_learning_rate",
    "uncertainty_learning_rate",
    "classifier_learning_rate",
)

@dataclass
class UFMSAConfig:
    roberta_name_or_path: str = "PATH_OR_HF_ID_FOR_ROBERTA"
    vit_name_or_path: str = "PATH_OR_HF_ID_FOR_VIT"
    image_root: str = "PATH_TO_IMAGE_ROOT"
    output_dir: str = "PATH_TO_OUTPUT_DIRECTORY"
    local_files_only: bool = False
    dataset_name: str = "weibo"
    max_text_length: int = 160

    num_labels: int = 2
    model_dim: int = 512
    num_attention_heads: int = 8
    dropout: float = 0.5

    image_size: int = 224
    frequency_patch_size: int = 28
    stockwell_frequency_bins: int = 16
    frequency_channels: int = 256

    num_experts: int = 3
    expert_hidden_dim: int = 512
    expert_dropout: float = 0.5
    moe_entropy_weight: float = 1.0e-3

    mc_samples: int = 20
    uncertainty_head_hidden_dim: int = 256
    alpha_init_logit: float = 0.0
    beta_init: float = 5.0
    tau_init: float = 0.35

    batch_size: int = 16
    max_epochs: int = 100
    early_stopping_patience: int = 10

    text_learning_rate: float = 1.0e-4
    visual_learning_rate: float = 1.0e-4
    frequency_learning_rate: float = 1.0e-4
    smca_learning_rate: float = 1.0e-4
    moe_learning_rate: float = 1.0e-4
    uncertainty_learning_rate: float = 1.0e-4
    classifier_learning_rate: float = 1.0e-4
    weight_decay: float = 0.15
    max_grad_norm: float = 1.0
    num_workers: int = 4

    def validate(self) -> None:
        if self.model_dim % self.num_attention_heads != 0:
            raise ValueError("model_dim must be divisible by num_attention_heads")
        if self.image_size % self.frequency_patch_size != 0:
            raise ValueError("image_size must be divisible by frequency_patch_size")
        if self.num_experts != 3:
            LOGGER.warning(
                "The manuscript's default configuration uses K=3 experts; current value is %d.",
                self.num_experts,
            )
        if self.mc_samples < 1:
            raise ValueError("mc_samples must be at least 1")
        if self.dataset_name not in {"weibo", "finefake", "pheme"}:
            raise ValueError("dataset_name must be one of: weibo, finefake, pheme")
        expected_length = 160 if self.dataset_name == "weibo" else 50
        if self.max_text_length != expected_length:
            raise ValueError(
                f"max_text_length must be {expected_length} for {self.dataset_name}"
            )

def set_global_seed(seed: int) -> None:

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        pass

@contextlib.contextmanager
def mc_dropout_mode(module: nn.Module) -> Iterator[None]:

    states = {submodule: submodule.training for submodule in module.modules()}
    module.eval()
    for submodule in module.modules():
        if isinstance(
            submodule,
            (
                nn.Dropout,
                nn.Dropout1d,
                nn.Dropout2d,
                nn.Dropout3d,
                nn.AlphaDropout,
                nn.FeatureAlphaDropout,
            ),
        ):
            submodule.train(True)
    try:
        yield
    finally:
        for submodule, state in states.items():
            submodule.train(state)

def _read_manifest(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".json"}:
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as file:
            if suffix == ".json":
                payload = json.load(file)
                if not isinstance(payload, list):
                    raise ValueError("A .json manifest must contain a list of records")
                rows = [dict(item) for item in payload]
            else:
                for line_number, line in enumerate(file, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(dict(json.loads(line)))
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid JSONL record at line {line_number} in {path}"
                        ) from exc
        return rows

    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            return [dict(row) for row in csv.DictReader(file)]

    raise ValueError("Manifest must be .jsonl, .json, or .csv")

def _manifest_record_key(row: Mapping[str, Any], source: Path, index: int) -> str:
    sample_id = str(row.get("sample_id", "")).strip()
    if sample_id:
        return sample_id
    text_value = str(row.get("text", ""))
    image_value = str(row.get("image_path", ""))
    if not text_value or not image_value:
        raise ValueError(f"Cannot identify record {index} in {source}")
    return json.dumps([text_value, image_value], ensure_ascii=False)

def validate_main_experiment_manifests(
    train_manifest: str,
    validation_manifest: str,
    test_manifest: str,
    tolerance: float = 0.01,
) -> Dict[str, Any]:
    if tolerance < 0:
        raise ValueError("split tolerance must be nonnegative")
    sources = {
        "train": Path(train_manifest),
        "validation": Path(validation_manifest),
        "test": Path(test_manifest),
    }
    rows = {name: _read_manifest(path) for name, path in sources.items()}
    if any(len(records) == 0 for records in rows.values()):
        raise ValueError("training, validation, and test manifests must be nonempty")
    keys: Dict[str, set[str]] = {}
    label_counts: Dict[str, Dict[int, int]] = {}
    for split_name, records in rows.items():
        split_keys: List[str] = []
        counts = {0: 0, 1: 0}
        for index, row in enumerate(records):
            label = int(row.get("label", -1))
            if label not in counts:
                raise ValueError(
                    f"Record {index} in {sources[split_name]} has a non-binary label"
                )
            counts[label] += 1
            split_keys.append(
                _manifest_record_key(row, sources[split_name], index)
            )
        if len(split_keys) != len(set(split_keys)):
            raise ValueError(f"Duplicate samples were found in the {split_name} split")
        if counts[0] == 0 or counts[1] == 0:
            raise ValueError(f"Both classes must occur in the {split_name} split")
        keys[split_name] = set(split_keys)
        label_counts[split_name] = counts
    if keys["train"] & keys["validation"]:
        raise ValueError("Training and validation manifests overlap")
    if keys["train"] & keys["test"]:
        raise ValueError("Training and test manifests overlap")
    if keys["validation"] & keys["test"]:
        raise ValueError("Validation and test manifests overlap")
    targets = {"train": 0.72, "validation": 0.08, "test": 0.20}
    total = sum(len(records) for records in rows.values())
    proportions = {
        name: len(records) / total for name, records in rows.items()
    }
    for name, target in targets.items():
        if abs(proportions[name] - target) > tolerance:
            raise ValueError(
                f"{name} proportion {proportions[name]:.6f} does not match {target:.2f}"
            )
    class_totals = {
        label: sum(label_counts[name][label] for name in rows)
        for label in (0, 1)
    }
    for label in (0, 1):
        for name, target in targets.items():
            class_proportion = label_counts[name][label] / class_totals[label]
            if abs(class_proportion - target) > tolerance:
                raise ValueError(
                    f"Class {label} is not stratified in the {name} split"
                )
    return {
        "total": total,
        "split_sizes": {name: len(records) for name, records in rows.items()},
        "split_proportions": proportions,
        "label_counts": label_counts,
    }

class MultimodalManifestDataset(Dataset):

    REQUIRED_FIELDS = ("text", "image_path", "label")

    def __init__(
        self,
        manifest_path: str,
        tokenizer: Any,
        image_processor: Any,
        image_root: str,
        max_text_length: int,
        image_size: int = 224,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.rows = _read_manifest(self.manifest_path)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_root = Path(image_root)
        self.max_text_length = max_text_length
        self.image_size = image_size

        for index, row in enumerate(self.rows):
            missing = [key for key in self.REQUIRED_FIELDS if key not in row]
            if missing:
                raise ValueError(
                    f"Manifest record {index} is missing required fields: {missing}"
                )

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_image_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        if not path.is_absolute():
            path = self.image_root / path
        return path

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        text = str(row["text"])
        image_path = self._resolve_image_path(str(row["image_path"]))
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
            vit_inputs = self.image_processor(
                images=image,
                size={"height": self.image_size, "width": self.image_size},
                return_tensors="pt",
            )
            pixel_values = vit_inputs["pixel_values"].squeeze(0)

            gray = image.convert("L").resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
            gray_array = np.asarray(gray, dtype=np.float32) / 255.0
            frequency_image = torch.from_numpy(gray_array).unsqueeze(0)

        text_inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )

        output: Dict[str, Any] = {
            "input_ids": text_inputs["input_ids"].squeeze(0),
            "attention_mask": text_inputs["attention_mask"].squeeze(0),
            "pixel_values": pixel_values,
            "frequency_image": frequency_image,
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "sample_id": str(row.get("sample_id", index)),
            "domain": str(row.get("domain", "")),
        }
        return output

def build_dataloader(
    manifest_path: str,
    tokenizer: Any,
    image_processor: Any,
    image_root: str,
    max_text_length: int,
    image_size: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    dataset = MultimodalManifestDataset(
        manifest_path=manifest_path,
        tokenizer=tokenizer,
        image_processor=image_processor,
        image_root=image_root,
        max_text_length=max_text_length,
        image_size=image_size,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        persistent_workers=num_workers > 0,
    )

class TextEncoder(nn.Module):

    def __init__(self, config: UFMSAConfig) -> None:
        super().__init__()
        if AutoModel is None:
            raise ImportError(
                "The `transformers` package is required. Install it with `pip install transformers`."
            )
        self.backbone = AutoModel.from_pretrained(
            config.roberta_name_or_path,
            local_files_only=config.local_files_only,
        )
        hidden_size = int(self.backbone.config.hidden_size)
        self.projection = nn.Linear(hidden_size, config.model_dim)
        self.norm = nn.LayerNorm(config.model_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "last_hidden_state"):
            cls_feature = outputs.last_hidden_state[:, 0]
        else:
            cls_feature = outputs[0][:, 0]
        feature = F.relu(self.projection(cls_feature))
        feature = self.norm(feature)
        return self.dropout(feature)

class VisualEncoder(nn.Module):

    def __init__(self, config: UFMSAConfig) -> None:
        super().__init__()
        if AutoModel is None:
            raise ImportError(
                "The `transformers` package is required. Install it with `pip install transformers`."
            )
        self.backbone = AutoModel.from_pretrained(
            config.vit_name_or_path,
            local_files_only=config.local_files_only,
        )
        hidden_size = int(self.backbone.config.hidden_size)
        self.projection = nn.Linear(hidden_size, config.model_dim)
        self.norm = nn.BatchNorm1d(config.model_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=pixel_values)
        if hasattr(outputs, "last_hidden_state"):
            cls_feature = outputs.last_hidden_state[:, 0]
        else:
            cls_feature = outputs[0][:, 0]
        feature = F.relu(self.projection(cls_feature))
        feature = self.norm(feature)
        return self.dropout(feature)

class DiscreteStockwellTransform1D(nn.Module):

    def __init__(self, signal_length: int, frequency_bins: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        if signal_length < 4:
            raise ValueError("signal_length must be at least 4")
        if frequency_bins < 1:
            raise ValueError("frequency_bins must be positive")
        self.signal_length = signal_length
        self.frequency_bins = frequency_bins
        self.eps = eps

        max_bin = signal_length // 2
        selected = torch.linspace(1, max_bin, steps=frequency_bins).round().long()
        selected = torch.unique_consecutive(selected)
        if selected.numel() != frequency_bins:
            selected = torch.arange(1, frequency_bins + 1).clamp(max=max_bin)
        self.register_buffer("selected_bins", selected, persistent=True)

        normalized_frequency = torch.fft.fftfreq(signal_length)
        windows: List[torch.Tensor] = []
        for bin_index in selected.tolist():
            center_frequency = max(float(bin_index) / signal_length, eps)
            gaussian = torch.exp(
                -2.0
                * math.pi**2
                * normalized_frequency.square()
                / (center_frequency**2 + eps)
            )
            windows.append(gaussian)
        self.register_buffer("gaussian_windows", torch.stack(windows, dim=0), persistent=True)

    def forward(self, signals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if signals.ndim != 3 or signals.shape[-1] != self.signal_length:
            raise ValueError(
                f"Expected [B, P, {self.signal_length}], got {tuple(signals.shape)}"
            )

        spectrum = torch.fft.fft(signals, dim=-1)
        coefficients: List[torch.Tensor] = []
        for window_index, bin_index in enumerate(self.selected_bins.tolist()):
            shifted_spectrum = torch.roll(spectrum, shifts=-bin_index, dims=-1)
            localized = torch.fft.ifft(
                shifted_spectrum * self.gaussian_windows[window_index], dim=-1
            )
            coefficients.append(localized)
        stockwell = torch.stack(coefficients, dim=2)

        magnitude = torch.log1p(torch.abs(stockwell))
        phase = torch.angle(stockwell) / math.pi

        magnitude = self._standardize(magnitude)
        phase = self._standardize(phase)
        return magnitude, phase

    def _standardize(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = tensor.mean(dim=-1, keepdim=True)
        std = tensor.std(dim=-1, keepdim=True, unbiased=False).clamp_min(self.eps)
        return (tensor - mean) / std

class FrequencyEncoder(nn.Module):

    def __init__(self, config: UFMSAConfig) -> None:
        super().__init__()
        patch_signal_length = config.frequency_patch_size**2
        self.image_size = config.image_size
        self.patch_size = config.frequency_patch_size
        self.num_patches_per_side = config.image_size // config.frequency_patch_size
        self.num_patches = self.num_patches_per_side**2

        self.stockwell = DiscreteStockwellTransform1D(
            signal_length=patch_signal_length,
            frequency_bins=config.stockwell_frequency_bins,
        )
        in_channels = 2 * config.stockwell_frequency_bins
        self.local_encoder = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, config.frequency_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(config.frequency_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.projection = nn.Linear(config.frequency_channels, config.model_dim)
        self.norm = nn.LayerNorm(config.model_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, grayscale_images: torch.Tensor) -> torch.Tensor:
        if grayscale_images.ndim != 4 or grayscale_images.shape[1] != 1:
            raise ValueError("frequency_image must have shape [B, 1, H, W]")
        if grayscale_images.shape[-2:] != (self.image_size, self.image_size):
            grayscale_images = F.interpolate(
                grayscale_images,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

        patches = F.unfold(
            grayscale_images,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        patches = patches.transpose(1, 2)

        with torch.no_grad():
            magnitude, phase = self.stockwell(patches)
        local_input = torch.cat([magnitude, phase], dim=2)
        batch_size, patch_count, channels, length = local_input.shape
        local_input = local_input.reshape(batch_size * patch_count, channels, length)

        local_features = self.local_encoder(local_input).squeeze(-1)
        local_features = local_features.reshape(batch_size, patch_count, -1)
        pooled = local_features.mean(dim=1)
        feature = self.projection(pooled)
        feature = self.norm(feature)
        return self.dropout(feature)

class DirectionalFeatureAttention(nn.Module):

    def __init__(self, model_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.output_projection = nn.Linear(model_dim, model_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(
        self,
        projected_query: torch.Tensor,
        projected_key: torch.Tensor,
        projected_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = projected_query.shape[0]
        query = projected_query.reshape(batch_size, self.num_heads, self.head_dim)
        key = projected_key.reshape(batch_size, self.num_heads, self.head_dim)
        value = projected_value.reshape(batch_size, self.num_heads, self.head_dim)
        scores = torch.einsum("bhi,bhj->bhij", query, key) / math.sqrt(self.head_dim)
        weights = F.softmax(scores, dim=-1)
        weights = self.attention_dropout(weights)
        context = torch.einsum("bhij,bhj->bhi", weights, value)
        context = context.reshape(batch_size, self.model_dim)
        output = self.output_dropout(self.output_projection(context))
        return output, weights

class SymmetricMultimodalCoAttention(nn.Module):

    def __init__(self, config: UFMSAConfig) -> None:
        super().__init__()
        modalities = ("t", "d", "v")
        directions = ("td", "tv", "dt", "dv", "vt", "vd")
        self.query_projections = nn.ModuleDict(
            {name: nn.Linear(config.model_dim, config.model_dim) for name in modalities}
        )
        self.key_projections = nn.ModuleDict(
            {name: nn.Linear(config.model_dim, config.model_dim) for name in modalities}
        )
        self.value_projections = nn.ModuleDict(
            {name: nn.Linear(config.model_dim, config.model_dim) for name in modalities}
        )
        self.directional_attention = nn.ModuleDict(
            {
                name: DirectionalFeatureAttention(
                    config.model_dim,
                    config.num_attention_heads,
                    config.dropout,
                )
                for name in directions
            }
        )
        self.text_norm = nn.LayerNorm(config.model_dim)
        self.frequency_norm = nn.LayerNorm(config.model_dim)
        self.visual_norm = nn.LayerNorm(config.model_dim)

    def forward(
        self,
        text_feature: torch.Tensor,
        frequency_feature: torch.Tensor,
        visual_feature: torch.Tensor,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        features = {
            "t": text_feature,
            "d": frequency_feature,
            "v": visual_feature,
        }
        queries = {
            name: self.query_projections[name](feature)
            for name, feature in features.items()
        }
        keys = {
            name: self.key_projections[name](feature)
            for name, feature in features.items()
        }
        values = {
            name: self.value_projections[name](feature)
            for name, feature in features.items()
        }
        outputs: Dict[str, torch.Tensor] = {}
        weights: Dict[str, torch.Tensor] = {}
        for direction in ("td", "tv", "dt", "dv", "vt", "vd"):
            query_modality, source_modality = direction
            output, attention = self.directional_attention[direction](
                queries[query_modality],
                keys[source_modality],
                values[source_modality],
            )
            outputs[direction] = output
            weights[direction] = attention

        enhanced_text = self.text_norm(text_feature + outputs["td"] + outputs["tv"])
        enhanced_frequency = self.frequency_norm(
            frequency_feature + outputs["dt"] + outputs["dv"]
        )
        enhanced_visual = self.visual_norm(
            visual_feature + outputs["vt"] + outputs["vd"]
        )
        attention_map = {
            "t_to_d": weights["td"],
            "t_to_v": weights["tv"],
            "d_to_t": weights["dt"],
            "d_to_v": weights["dv"],
            "v_to_t": weights["vt"],
            "v_to_d": weights["vd"],
        }
        return (enhanced_text, enhanced_frequency, enhanced_visual), attention_map

class FeedForwardExpert(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, unified_input: torch.Tensor) -> torch.Tensor:
        return self.network(unified_input)

class SampleAdaptiveMoE(nn.Module):

    def __init__(self, config: UFMSAConfig) -> None:
        super().__init__()
        self.model_dim = config.model_dim
        self.num_experts = config.num_experts
        self.input_dim = config.model_dim * 3 + 3

        self.gate = nn.Linear(self.input_dim, config.num_experts)
        self.experts = nn.ModuleList(
            [
                FeedForwardExpert(
                    input_dim=self.input_dim,
                    hidden_dim=config.expert_hidden_dim,
                    output_dim=config.model_dim,
                    dropout=config.expert_dropout,
                )
                for _ in range(config.num_experts)
            ]
        )
        self.residual_projection = nn.Linear(self.input_dim, config.model_dim)
        self.output_norm = nn.LayerNorm(config.model_dim)

    @staticmethod
    def _cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return F.cosine_similarity(left, right, dim=-1).unsqueeze(-1)

    def forward(
        self,
        enhanced_text: torch.Tensor,
        enhanced_frequency: torch.Tensor,
        enhanced_visual: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        similarity_td = self._cosine(enhanced_text, enhanced_frequency)
        similarity_tv = self._cosine(enhanced_text, enhanced_visual)
        similarity_dv = self._cosine(enhanced_frequency, enhanced_visual)

        unified_input = torch.cat(
            [
                enhanced_text,
                enhanced_frequency,
                enhanced_visual,
                similarity_td,
                similarity_tv,
                similarity_dv,
            ],
            dim=-1,
        )
        if unified_input.shape[-1] != self.input_dim:
            raise RuntimeError(
                f"MoE input dimension must be {self.input_dim}, got {unified_input.shape[-1]}"
            )

        gate_weights = F.softmax(self.gate(unified_input), dim=-1)
        expert_outputs = torch.stack(
            [expert(unified_input) for expert in self.experts], dim=1
        )
        weighted_experts = torch.sum(gate_weights.unsqueeze(-1) * expert_outputs, dim=1)
        residual = self.residual_projection(unified_input)
        moe_feature = self.output_norm(weighted_experts + residual)

        routing_entropy = -torch.sum(
            gate_weights * torch.log(gate_weights.clamp_min(1.0e-8)), dim=-1
        ).mean()

        details = {
            "unified_input": unified_input,
            "gate_weights": gate_weights,
            "expert_outputs": expert_outputs,
            "routing_entropy": routing_entropy,
            "similarity_td": similarity_td.squeeze(-1),
            "similarity_tv": similarity_tv.squeeze(-1),
            "similarity_dv": similarity_dv.squeeze(-1),
        }
        return moe_feature, details

class ModalityDecisionHead(nn.Module):

    def __init__(self, config: UFMSAConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.model_dim, config.uncertainty_head_hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(config.uncertainty_head_hidden_dim, config.num_labels)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.fc1(feature))
        hidden = self.dropout(hidden)
        return self.fc2(hidden)

class HierarchicalUncertaintyFusion(nn.Module):

    def __init__(self, config: UFMSAConfig) -> None:
        super().__init__()
        self.text_head = ModalityDecisionHead(config)
        self.frequency_head = ModalityDecisionHead(config)
        self.visual_head = ModalityDecisionHead(config)
        self.alpha_logit = nn.Parameter(torch.tensor(config.alpha_init_logit, dtype=torch.float32))
        self.eps = 1.0e-8

    @staticmethod
   def _feature_uncertainty(samples: torch.Tensor) -> torch.Tensor:
       sample_mean = samples.mean(dim=0, keepdim=True)
       turn (samples - sample_mean).square().mean(dim=-1).mean(dim=0)

    def _decision_uncertainty(
        self, samples: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        text_logits = torch.stack(
            [self.text_head(samples[index, :, 0]) for index in range(samples.shape[0])],
            dim=0,
        )
        frequency_logits = torch.stack(
            [
                self.frequency_head(samples[index, :, 1])
                for index in range(samples.shape[0])
            ],
            dim=0,
        )
        visual_logits = torch.stack(
            [self.visual_head(samples[index, :, 2]) for index in range(samples.shape[0])],
            dim=0,
        )
        logits = torch.stack([text_logits, frequency_logits, visual_logits], dim=2)
        probabilities = F.softmax(logits, dim=-1)
        mean_probability = probabilities.mean(dim=0)
        predictive_entropy = -torch.sum(
            mean_probability * torch.log(mean_probability.clamp_min(self.eps)), dim=-1
        )
        return predictive_entropy, mean_probability

    def forward(
        self,
        deterministic_features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        mc_samples: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if mc_samples.ndim != 4 or mc_samples.shape[2] != 3:
            raise ValueError("mc_samples must have shape [T, B, 3, D]")

        feature_uncertainty = self._feature_uncertainty(mc_samples)
        decision_uncertainty, mean_probability = self._decision_uncertainty(mc_samples)

        alpha = torch.sigmoid(self.alpha_logit)
        total_uncertainty = (
            alpha * feature_uncertainty + (1.0 - alpha) * decision_uncertainty
        )
        confidence_weights = F.softmax(-total_uncertainty, dim=-1)

        deterministic_stack = torch.stack(deterministic_features, dim=1)
        hufn_feature = torch.sum(confidence_weights.unsqueeze(-1) * deterministic_stack, dim=1)
        global_uncertainty = torch.sum(
            confidence_weights * decision_uncertainty, dim=-1, keepdim=True
        )

        details = {
            "feature_uncertainty": feature_uncertainty,
            "decision_uncertainty": decision_uncertainty,
            "total_uncertainty": total_uncertainty,
            "confidence_weights": confidence_weights,
            "mean_modality_probability": mean_probability,
            "alpha": alpha,
        }
        return hufn_feature, global_uncertainty, details

class UncertaintyAdaptiveFusion(nn.Module):
    def __init__(self, config: UFMSAConfig) -> None:
        super().__init__()
        beta_raw = math.log(math.exp(config.beta_init) - 1.0)
        self.beta_raw = nn.Parameter(torch.tensor(beta_raw, dtype=torch.float32))
        self.tau = nn.Parameter(torch.tensor(config.tau_init, dtype=torch.float32))

    def forward(
        self,
        moe_feature: torch.Tensor,
        hufn_feature: torch.Tensor,
        global_uncertainty: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        beta = F.softplus(self.beta_raw) + 1.0e-6
        fusion_weight = torch.sigmoid(beta * (self.tau - global_uncertainty))
        final_feature = fusion_weight * moe_feature + (1.0 - fusion_weight) * hufn_feature
        return final_feature, fusion_weight, {"beta": beta, "tau": self.tau}

class UFMSA(nn.Module):
    def __init__(self, config: UFMSAConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.text_encoder = TextEncoder(config)
        self.visual_encoder = VisualEncoder(config)
        self.frequency_encoder = FrequencyEncoder(config)
        self.smca = SymmetricMultimodalCoAttention(config)
        self.moe = SampleAdaptiveMoE(config)
        self.hufn = HierarchicalUncertaintyFusion(config)
        self.uaf = UncertaintyAdaptiveFusion(config)
        self.classifier = nn.Sequential(
            nn.Linear(config.model_dim, config.uncertainty_head_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.uncertainty_head_hidden_dim, config.num_labels),
        )

    def _encode_modalities(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        frequency_image: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        text_feature = self.text_encoder(input_ids, attention_mask)
        frequency_feature = self.frequency_encoder(frequency_image)
        visual_feature = self.visual_encoder(pixel_values)
        return text_feature, frequency_feature, visual_feature

    def _smca_once(
        self,
        modality_features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        return self.smca(*modality_features)

    def _sample_smca_features(
        self,
        enhanced_features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        sampled_features: List[torch.Tensor] = []
        for _ in range(self.config.mc_samples):
            sampled_features.append(
                torch.stack(
                    [
                        F.dropout(
                            feature,
                            p=self.config.dropout,
                            training=True,
                        )
                        for feature in enhanced_features
                    ],
                    dim=1,
                )
            )
        return torch.stack(sampled_features, dim=0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        frequency_image: torch.Tensor,
        return_details: bool = True,
    ) -> Dict[str, torch.Tensor]:
        modality_features = self._encode_modalities(
            input_ids, attention_mask, pixel_values, frequency_image
        )
        enhanced_features, attention_map = self._smca_once(modality_features)
        moe_feature, moe_details = self.moe(*enhanced_features)

        mc_samples = self._sample_smca_features(enhanced_features)

        with mc_dropout_mode(self.hufn):
            hufn_feature, global_uncertainty, hufn_details = self.hufn(
                deterministic_features=enhanced_features,
                mc_samples=mc_samples,
            )

        final_feature, fusion_weight, uaf_details = self.uaf(
            moe_feature=moe_feature,
            hufn_feature=hufn_feature,
            global_uncertainty=global_uncertainty,
        )
        logits = self.classifier(final_feature)
        probabilities = F.softmax(logits, dim=-1)

        output: Dict[str, torch.Tensor] = {
            "logits": logits,
            "probabilities": probabilities,
            "final_feature": final_feature,
            "moe_feature": moe_feature,
            "hufn_feature": hufn_feature,
            "global_uncertainty": global_uncertainty,
            "fusion_weight": fusion_weight,
            "routing_entropy": moe_details["routing_entropy"],
        }
        if return_details:
            output.update(
                {
                    "gate_weights": moe_details["gate_weights"],
                    "similarity_td": moe_details["similarity_td"],
                    "similarity_tv": moe_details["similarity_tv"],
                    "similarity_dv": moe_details["similarity_dv"],
                    "feature_uncertainty": hufn_details["feature_uncertainty"],
                    "decision_uncertainty": hufn_details["decision_uncertainty"],
                    "total_uncertainty": hufn_details["total_uncertainty"],
                    "confidence_weights": hufn_details["confidence_weights"],
                    "alpha": hufn_details["alpha"],
                    "beta": uaf_details["beta"],
                    "tau": uaf_details["tau"],
                    "smca_t_to_d_weights": attention_map["t_to_d"],
                    "smca_t_to_v_weights": attention_map["t_to_v"],
                    "smca_d_to_t_weights": attention_map["d_to_t"],
                    "smca_d_to_v_weights": attention_map["d_to_v"],
                    "smca_v_to_t_weights": attention_map["v_to_t"],
                    "smca_v_to_d_weights": attention_map["v_to_d"],
                }
            )
        return output

    def compute_loss(
        self, model_output: Mapping[str, torch.Tensor], labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        classification_loss = F.cross_entropy(model_output["logits"], labels)
        routing_entropy = model_output["routing_entropy"]
        total_loss = (
            classification_loss
            - self.config.moe_entropy_weight * routing_entropy
        )
        return total_loss, {
            "classification_loss": classification_loss.detach(),
            "routing_entropy": routing_entropy.detach(),
            "total_loss": total_loss.detach(),
        }

@dataclass
class Metrics:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    loss: float

def _move_batch(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved

def evaluate(
    model: UFMSA,
    loader: DataLoader,
    device: torch.device,
    collect_details: bool = True,
) -> Tuple[Metrics, Dict[str, Any]]:
    model.eval()
    losses: List[float] = []
    labels_all: List[int] = []
    predictions_all: List[int] = []
    sample_records: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                frequency_image=batch["frequency_image"],
                return_details=True,
            )
            loss, _ = model.compute_loss(output, batch["label"])
            losses.append(float(loss.item()))
            predictions = output["logits"].argmax(dim=-1)
            labels_all.extend(batch["label"].cpu().tolist())
            predictions_all.extend(predictions.cpu().tolist())

            if collect_details:
                for index, sample_id in enumerate(batch["sample_id"]):
                    sample_records.append(
                        {
                            "sample_id": str(sample_id),
                            "label": int(batch["label"][index].item()),
                            "prediction": int(predictions[index].item()),
                            "probabilities": output["probabilities"][index].cpu().tolist(),
                            "gate_weights": output["gate_weights"][index].cpu().tolist(),
                            "similarity_td": float(output["similarity_td"][index].item()),
                            "similarity_tv": float(output["similarity_tv"][index].item()),
                            "similarity_dv": float(output["similarity_dv"][index].item()),
                            "confidence_weights": output["confidence_weights"][index].cpu().tolist(),
                            "feature_uncertainty": output["feature_uncertainty"][index].cpu().tolist(),
                            "decision_uncertainty": output["decision_uncertainty"][index].cpu().tolist(),
                            "global_uncertainty": float(output["global_uncertainty"][index].item()),
                            "fusion_weight": float(output["fusion_weight"][index].item()),
                        }
                    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_all,
        predictions_all,
        average="macro",
        zero_division=0,
    )
    metrics = Metrics(
        accuracy=float(accuracy_score(labels_all, predictions_all)),
        macro_precision=float(precision),
        macro_recall=float(recall),
        macro_f1=float(f1),
        loss=float(np.mean(losses)) if losses else float("nan"),
    )
    return metrics, {"samples": sample_records}

def build_optimizer(model: UFMSA, config: UFMSAConfig) -> torch.optim.Optimizer:

    try:
        from adabelief_pytorch import AdaBelief
    except ImportError as exc:
        raise ImportError(
            "UFMSA uses AdaBelief as specified in the manuscript. Install it with "
            "`pip install adabelief-pytorch`."
        ) from exc

    grouped: Dict[str, List[nn.Parameter]] = {
        "text": [],
        "visual": [],
        "frequency": [],
        "smca": [],
        "moe": [],
        "uncertainty": [],
        "classifier": [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("text_encoder"):
            grouped["text"].append(parameter)
        elif name.startswith("visual_encoder"):
            grouped["visual"].append(parameter)
        elif name.startswith("frequency_encoder"):
            grouped["frequency"].append(parameter)
        elif name.startswith("smca"):
            grouped["smca"].append(parameter)
        elif name.startswith("moe"):
            grouped["moe"].append(parameter)
        elif name.startswith("hufn") or name.startswith("uaf"):
            grouped["uncertainty"].append(parameter)
        elif name.startswith("classifier"):
            grouped["classifier"].append(parameter)
        else:
            raise RuntimeError(f"Unassigned trainable parameter: {name}")

    learning_rates = {
        "text": config.text_learning_rate,
        "visual": config.visual_learning_rate,
        "frequency": config.frequency_learning_rate,
        "smca": config.smca_learning_rate,
        "moe": config.moe_learning_rate,
        "uncertainty": config.uncertainty_learning_rate,
        "classifier": config.classifier_learning_rate,
    }
    parameter_groups = [
        {
            "params": parameters,
            "lr": learning_rates[group_name],
            "weight_decay": config.weight_decay,
            "name": group_name,
        }
        for group_name, parameters in grouped.items()
        if parameters
    ]
    return AdaBelief(
        parameter_groups,
        lr=1.0e-4,
        eps=1.0e-8,
        betas=(0.9, 0.999),
        weight_decouple=True,
        rectify=True,
        print_change_log=False,
    )

def train_epoch(
    model: UFMSA,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float,
) -> float:
    model.train()
    losses: List[float] = []
    for batch in loader:
        batch = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            frequency_image=batch["frequency_image"],
            return_details=False,
        )
        loss, _ = model.compute_loss(output, batch["label"])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss encountered: {loss.item()}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")

def load_processors(config: UFMSAConfig) -> Tuple[Any, Any]:
    if AutoTokenizer is None or AutoImageProcessor is None:
        raise ImportError(
            "The `transformers` package is required. Install it with `pip install transformers`."
        )
    tokenizer = AutoTokenizer.from_pretrained(
        config.roberta_name_or_path,
        local_files_only=config.local_files_only,
        use_fast=True,
    )
    image_processor = AutoImageProcessor.from_pretrained(
        config.vit_name_or_path,
        local_files_only=config.local_files_only,
    )
    return tokenizer, image_processor

def config_with_learning_rates(
    base_config: UFMSAConfig,
    learning_rates: Mapping[str, float],
) -> UFMSAConfig:
    payload = asdict(base_config)
    for field in LEARNING_RATE_FIELDS:
        if field not in learning_rates:
            raise KeyError(f"Missing tuned parameter: {field}")
        payload[field] = float(learning_rates[field])
    selected = UFMSAConfig(**payload)
    selected.validate()
    return selected

def ray_tune_trainable(
    trial_parameters: Mapping[str, float],
    base_config_payload: Mapping[str, Any],
    train_manifest: str,
    validation_manifest: str,
    tuning_seed: int,
) -> None:
    if tune is None:
        raise ImportError("Ray Tune is required. Install it with `pip install 'ray[tune]'`.")
    base_config = UFMSAConfig(**dict(base_config_payload))
    trial_config = config_with_learning_rates(base_config, trial_parameters)
    set_global_seed(tuning_seed)
    tokenizer, image_processor = load_processors(trial_config)
    train_loader = build_dataloader(
        manifest_path=train_manifest,
        tokenizer=tokenizer,
        image_processor=image_processor,
        image_root=trial_config.image_root,
        max_text_length=trial_config.max_text_length,
        image_size=trial_config.image_size,
        batch_size=trial_config.batch_size,
        shuffle=True,
        num_workers=trial_config.num_workers,
        seed=tuning_seed,
    )
    validation_loader = build_dataloader(
        manifest_path=validation_manifest,
        tokenizer=tokenizer,
        image_processor=image_processor,
        image_root=trial_config.image_root,
        max_text_length=trial_config.max_text_length,
        image_size=trial_config.image_size,
        batch_size=trial_config.batch_size,
        shuffle=False,
        num_workers=trial_config.num_workers,
        seed=tuning_seed,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UFMSA(trial_config).to(device)
    optimizer = build_optimizer(model, trial_config)
    best_validation_f1 = -float("inf")
    epochs_without_improvement = 0
    for epoch in range(1, trial_config.max_epochs + 1):
        training_loss = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            max_grad_norm=trial_config.max_grad_norm,
        )
        validation_metrics, _ = evaluate(
            model,
            validation_loader,
            device,
            collect_details=False,
        )
        if validation_metrics.macro_f1 > best_validation_f1:
            best_validation_f1 = validation_metrics.macro_f1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        tune.report(
            {
                "epoch": epoch,
                "training_loss": training_loss,
                "validation_loss": validation_metrics.loss,
                "validation_accuracy": validation_metrics.accuracy,
                "validation_macro_precision": validation_metrics.macro_precision,
                "validation_macro_recall": validation_metrics.macro_recall,
                "validation_macro_f1": validation_metrics.macro_f1,
                "best_validation_macro_f1": best_validation_f1,
            }
        )
        if epochs_without_improvement >= trial_config.early_stopping_patience:
            break

def run_ray_tune(
    config: UFMSAConfig,
    train_manifest: str,
    validation_manifest: str,
    num_samples: int,
    minimum_learning_rate: float,
    maximum_learning_rate: float,
    tuning_seed: int,
    cpus_per_trial: float,
    gpus_per_trial: float,
    max_concurrent_trials: int,
    experiment_name: str,
) -> Dict[str, float]:
    if tune is None or BasicVariantGenerator is None:
        raise ImportError("Ray Tune is required. Install it with `pip install 'ray[tune]'`.")
    if num_samples < 1:
        raise ValueError("ray_tune_num_samples must be at least 1")
    if minimum_learning_rate <= 0 or maximum_learning_rate <= minimum_learning_rate:
        raise ValueError("Ray Tune learning-rate bounds are invalid")
    if cpus_per_trial <= 0 or gpus_per_trial < 0:
        raise ValueError("Ray Tune resource values are invalid")
    if max_concurrent_trials < 1:
        raise ValueError("ray_tune_max_concurrent_trials must be at least 1")
    search_space = {
        field: tune.loguniform(minimum_learning_rate, maximum_learning_rate)
        for field in LEARNING_RATE_FIELDS
    }
    trainable = tune.with_parameters(
        ray_tune_trainable,
        base_config_payload=asdict(config),
        train_manifest=train_manifest,
        validation_manifest=validation_manifest,
        tuning_seed=tuning_seed,
    )
    trainable = tune.with_resources(
        trainable,
        resources={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
    )
    search_algorithm = BasicVariantGenerator(
        random_state=tuning_seed,
        max_concurrent=max_concurrent_trials,
    )
    storage_path = str((Path(config.output_dir).resolve() / "ray_tune").resolve())
    tuner = tune.Tuner(
        trainable,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="best_validation_macro_f1",
            mode="max",
            num_samples=num_samples,
            search_alg=search_algorithm,
            max_concurrent_trials=max_concurrent_trials,
        ),
        run_config=tune.RunConfig(
            name=experiment_name,
            storage_path=storage_path,
        ),
    )
    result_grid = tuner.fit()
    if result_grid.num_terminated == 0:
        raise RuntimeError("Ray Tune did not complete any successful trial")
    best_result = result_grid.get_best_result(
        metric="best_validation_macro_f1",
        mode="max",
        scope="all",
    )
    best_learning_rates = {
        field: float(best_result.config[field])
        for field in LEARNING_RATE_FIELDS
    }
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "ray_tune_best_learning_rates.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            {
                "best_validation_macro_f1": float(
                    best_result.metrics["best_validation_macro_f1"]
                ),
                "tuning_seed": tuning_seed,
                "search_range": [minimum_learning_rate, maximum_learning_rate],
                "num_samples": num_samples,
                "learning_rates": best_learning_rates,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    result_grid.get_dataframe().to_csv(
        output_dir / "ray_tune_trials.csv",
        index=False,
    )
    return best_learning_rates

def train_one_seed(
    config: UFMSAConfig,
    seed: int,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> Tuple[Metrics, Path]:
    set_global_seed(seed)
    model = UFMSA(config).to(device)
    optimizer = build_optimizer(model, config)

    run_dir = Path(config.output_dir) / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best_validation_macro_f1.pt"

    best_validation_f1 = -float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, config.max_epochs + 1):
        training_loss = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            max_grad_norm=config.max_grad_norm,
        )
        validation_metrics, _ = evaluate(
            model,
            validation_loader,
            device,
            collect_details=False,
        )
        LOGGER.info(
            "seed=%d epoch=%d train_loss=%.6f val_macro_f1=%.6f",
            seed,
            epoch,
            training_loss,
            validation_metrics.macro_f1,
        )

        if validation_metrics.macro_f1 > best_validation_f1:
            best_validation_f1 = validation_metrics.macro_f1
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "seed": seed,
                    "validation_macro_f1": best_validation_f1,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                LOGGER.info("Early stopping for seed %d at epoch %d", seed, epoch)
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics, test_details = evaluate(
        model, test_loader, device, collect_details=True
    )

    with (run_dir / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(asdict(test_metrics), file, ensure_ascii=False, indent=2)
    with (run_dir / "test_predictions_and_analysis.jsonl").open(
        "w", encoding="utf-8"
    ) as file:
        for record in test_details["samples"]:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return test_metrics, checkpoint_path

def summarize_runs(run_metrics: Mapping[int, Metrics]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    metric_names = ("accuracy", "macro_precision", "macro_recall", "macro_f1", "loss")
    for metric_name in metric_names:
        values = np.asarray(
            [getattr(metrics, metric_name) for metrics in run_metrics.values()], dtype=np.float64
        )
        summary[metric_name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    return summary

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the UFMSA main experiment")
    parser.add_argument("--train-manifest", required=True, help="PATH_TO_TRAIN_MANIFEST")
    parser.add_argument("--validation-manifest", required=True, help="PATH_TO_VALIDATION_MANIFEST")
    parser.add_argument("--test-manifest", required=True, help="PATH_TO_TEST_MANIFEST")
    parser.add_argument(
        "--image-root",
        default=os.environ.get("UFMSA_IMAGE_ROOT", "PATH_TO_IMAGE_ROOT"),
    )
    parser.add_argument(
        "--roberta-model",
        default=os.environ.get("UFMSA_ROBERTA_MODEL", "PATH_OR_HF_ID_FOR_ROBERTA"),
    )
    parser.add_argument(
        "--vit-model",
        default=os.environ.get("UFMSA_VIT_MODEL", "PATH_OR_HF_ID_FOR_VIT"),
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("UFMSA_OUTPUT_DIR", "PATH_TO_OUTPUT_DIRECTORY"),
    )
    parser.add_argument(
        "--dataset-name",
        choices=("weibo", "finefake", "pheme"),
        required=True,
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(PAPER_SEEDS),
        help="Paper seeds: 13 21 43 2026 3407",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--split-tolerance", type=float, default=0.01)
    parser.add_argument("--ray-tune-num-samples", type=int, default=30)
    parser.add_argument("--ray-tune-min-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--ray-tune-max-learning-rate", type=float, default=1.0e-2)
    parser.add_argument("--ray-tune-seed", type=int, default=43)
    parser.add_argument("--ray-tune-cpus-per-trial", type=float, default=4.0)
    parser.add_argument(
        "--ray-tune-gpus-per-trial",
        type=float,
        default=1.0 if torch.cuda.is_available() else 0.0,
    )
    parser.add_argument("--ray-tune-max-concurrent-trials", type=int, default=1)
    parser.add_argument(
        "--ray-tune-experiment-name",
        default="ufmsa_module_learning_rate_search",
    )
    parser.add_argument("--skip-ray-tune", action="store_true")
    return parser.parse_args()

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()
    max_text_length = 160 if args.dataset_name == "weibo" else 50
    config = UFMSAConfig(
        roberta_name_or_path=args.roberta_model,
        vit_name_or_path=args.vit_model,
        image_root=args.image_root,
        output_dir=args.output_dir,
        local_files_only=args.local_files_only,
        dataset_name=args.dataset_name,
        max_text_length=max_text_length,
        num_workers=args.num_workers,
    )
    config.validate()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_summary = validate_main_experiment_manifests(
        train_manifest=args.train_manifest,
        validation_manifest=args.validation_manifest,
        test_manifest=args.test_manifest,
        tolerance=args.split_tolerance,
    )
    with (output_dir / "data_split_summary.json").open("w", encoding="utf-8") as file:
        json.dump(split_summary, file, ensure_ascii=False, indent=2)

    if not args.skip_ray_tune:
        best_learning_rates = run_ray_tune(
            config=config,
            train_manifest=args.train_manifest,
            validation_manifest=args.validation_manifest,
            num_samples=args.ray_tune_num_samples,
            minimum_learning_rate=args.ray_tune_min_learning_rate,
            maximum_learning_rate=args.ray_tune_max_learning_rate,
            tuning_seed=args.ray_tune_seed,
            cpus_per_trial=args.ray_tune_cpus_per_trial,
            gpus_per_trial=args.ray_tune_gpus_per_trial,
            max_concurrent_trials=args.ray_tune_max_concurrent_trials,
            experiment_name=f"{args.ray_tune_experiment_name}_{args.dataset_name}",
        )
        config = config_with_learning_rates(config, best_learning_rates)

    with (output_dir / "configuration.json").open("w", encoding="utf-8") as file:
        json.dump(asdict(config), file, ensure_ascii=False, indent=2)

    tokenizer, image_processor = load_processors(config)
    device = torch.device(args.device)
    run_metrics: Dict[int, Metrics] = {}
    for seed in args.seeds:
        train_loader = build_dataloader(
            manifest_path=args.train_manifest,
            tokenizer=tokenizer,
            image_processor=image_processor,
            image_root=config.image_root,
            max_text_length=config.max_text_length,
            image_size=config.image_size,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            seed=seed,
        )
        validation_loader = build_dataloader(
            manifest_path=args.validation_manifest,
            tokenizer=tokenizer,
            image_processor=image_processor,
            image_root=config.image_root,
            max_text_length=config.max_text_length,
            image_size=config.image_size,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            seed=seed,
        )
        test_loader = build_dataloader(
            manifest_path=args.test_manifest,
            tokenizer=tokenizer,
            image_processor=image_processor,
            image_root=config.image_root,
            max_text_length=config.max_text_length,
            image_size=config.image_size,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            seed=seed,
        )
        metrics, checkpoint_path = train_one_seed(
            config=config,
            seed=seed,
            train_loader=train_loader,
            validation_loader=validation_loader,
            test_loader=test_loader,
            device=device,
        )
        run_metrics[seed] = metrics
        LOGGER.info("seed=%d test=%s checkpoint=%s", seed, metrics, checkpoint_path)

    summary = summarize_runs(run_metrics)
    with (output_dir / "five_seed_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    LOGGER.info("Five-seed summary: %s", json.dumps(summary, ensure_ascii=False))

if __name__ == "__main__":
    main()
