from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from miocodec import MioCodecModel

logger = logging.getLogger(__name__)


@dataclass
class PresetEntry:
    preset_id: str
    path: Path


class MioCodecService:
    def __init__(
        self,
        model_id: str,
        device: str,
        presets_dir: Path,
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._presets_dir = presets_dir
        self._codec: MioCodecModel | None = None
        self._preset_cache: dict[str, torch.Tensor] = {}

    def load(self) -> None:
        logger.info("Loading MioCodec model: %s", self._model_id)
        codec = MioCodecModel.from_pretrained(self._model_id)
        codec = codec.eval().to(self._device)
        self._codec = codec

    @property
    def codec(self) -> MioCodecModel:
        if self._codec is None:
            raise RuntimeError("MioCodec has not been loaded.")
        return self._codec

    @property
    def sample_rate(self) -> int:
        return int(self.codec.config.sample_rate)

    def list_presets(self) -> list[str]:
        if not self._presets_dir.exists():
            return []
        presets = []
        for path in self._presets_dir.iterdir():
            if path.suffix.lower() in {".pt", ".npz"}:
                presets.append(path.stem)
        return sorted(set(presets))

    def load_preset_embedding(self, preset_id: str) -> torch.Tensor:
        if preset_id in self._preset_cache:
            return self._preset_cache[preset_id]
        entry = self._resolve_preset(preset_id)
        embedding = _load_embedding_from_path(entry.path)
        embedding = _prepare_embedding(embedding, _codec_device(self.codec))
        self._preset_cache[preset_id] = embedding
        return embedding

    def synthesize(
        self,
        tokens: list[int] | torch.Tensor,
        reference_waveform: torch.Tensor | None = None,
        global_embedding: torch.Tensor | None = None,
        target_audio_length: int | None = None,
    ) -> torch.Tensor:
        if reference_waveform is None and global_embedding is None:
            raise ValueError("Either reference_waveform or global_embedding is required.")
        device = _codec_device(self.codec)

        # Extract global embedding from reference waveform if provided
        if reference_waveform is not None:
            reference_waveform = reference_waveform.to(device=device, dtype=torch.float32)
            ref_features = self.codec.encode(reference_waveform, return_content=False, return_global=True)
            global_embedding = ref_features.global_embedding

        if isinstance(tokens, list):
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
        elif isinstance(tokens, torch.Tensor) and tokens.dtype != torch.long:
            tokens = tokens.long().to(device)
        elif isinstance(tokens, torch.Tensor):
            tokens = tokens.to(device)
        return self.codec.decode(
            global_embedding=global_embedding,
            content_token_indices=tokens,
            target_audio_length=target_audio_length,
        )

    def synthesize_batch(
        self,
        tokens_list: list[list[int]] | list[torch.Tensor],
        reference_waveform: torch.Tensor | None = None,
        global_embedding: torch.Tensor | None = None,
        target_audio_lengths: list[int] | None = None,
        padding_token_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if reference_waveform is None and global_embedding is None:
            raise ValueError("Either reference_waveform or global_embedding is required.")
        logger.debug("Synthesize batch: items=%d", len(tokens_list))

        device = _codec_device(self.codec)

        # Extract global embedding from reference waveform if provided
        if reference_waveform is not None:
            reference_waveform = reference_waveform.to(device=device, dtype=torch.float32)
            ref_features = self.codec.encode(reference_waveform, return_content=False, return_global=True)
            global_embedding = ref_features.global_embedding
        token_tensors = []
        content_lengths = []
        for tokens in tokens_list:
            if isinstance(tokens, list):
                t = torch.tensor(tokens, dtype=torch.long, device=device)
            elif isinstance(tokens, torch.Tensor):
                t = tokens.long().to(device)
            else:
                raise TypeError("Each element of tokens_list must be a torch.Tensor or list[int].")
            token_tensors.append(t)
            content_lengths.append(t.shape[0])
        logger.debug("Batch token lengths: %s", content_lengths)
        max_seq_len = max(content_lengths)
        batch_tokens = torch.full(
            (len(tokens_list), max_seq_len), padding_token_idx, dtype=torch.long, device=device
        )
        for i, t in enumerate(token_tensors):
            batch_tokens[i, : t.shape[0]] = t

        global_embedding = _prepare_embedding(global_embedding, device)
        if global_embedding.dim() == 1:
            global_embeddings = global_embedding.unsqueeze(0).expand(len(tokens_list), -1)
        elif global_embedding.dim() == 2:
            if global_embedding.shape[0] == 1:
                global_embeddings = global_embedding.expand(len(tokens_list), -1)
            elif global_embedding.shape[0] == len(tokens_list):
                global_embeddings = global_embedding
            else:
                raise ValueError("global_embedding batch size does not match tokens_list.")
        else:
            raise ValueError("global_embedding must be 1D or 2D tensor.")

        return self.codec.decode_batch(
            global_embeddings=global_embeddings,
            content_token_indices=batch_tokens,
            content_lengths=content_lengths,
            target_audio_lengths=target_audio_lengths,
            padding_token_idx=padding_token_idx,
        )

    def _resolve_preset(self, preset_id: str) -> PresetEntry:
        preset_id = _sanitize_preset_id(preset_id)
        base_dir = self._presets_dir.resolve()
        candidates = [
            (base_dir / f"{preset_id}.pt").resolve(),
            (base_dir / f"{preset_id}.npz").resolve(),
        ]
        for path in candidates:
            if not _is_path_within(path, base_dir):
                logger.warning("Rejected preset path outside presets dir: %s", path)
                continue
            if path.exists():
                return PresetEntry(preset_id=preset_id, path=path)
        raise FileNotFoundError(f"Preset '{preset_id}' not found in {base_dir}.")


def _load_embedding_from_path(path: Path) -> Any:
    if path.suffix.lower() == ".pt":
        return torch.load(path, map_location="cpu", weights_only=True)
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        if "global_embedding" in data:
            return data["global_embedding"]
        if "embedding" in data:
            return data["embedding"]
        keys = list(data.keys())
        if keys:
            return data[keys[0]]
    raise ValueError(f"Unsupported preset format: {path}")


def _prepare_embedding(embedding: Any, device: torch.device | str) -> torch.Tensor:
    if isinstance(embedding, dict):
        if "global_embedding" in embedding:
            embedding = embedding["global_embedding"]
        elif "embedding" in embedding:
            embedding = embedding["embedding"]
    if isinstance(embedding, np.ndarray):
        embedding = torch.from_numpy(embedding)
    if not isinstance(embedding, torch.Tensor):
        embedding = torch.tensor(embedding)
    embedding = embedding.squeeze()
    if embedding.dim() != 1:
        embedding = embedding.flatten()
    return embedding.to(device)


def _codec_device(codec: MioCodecModel) -> torch.device:
    return next(codec.parameters()).device


def _sanitize_preset_id(preset_id: str) -> str:
    normalized = preset_id.strip()
    if not normalized:
        raise ValueError("Invalid preset_id: empty value.")
    if normalized in {".", ".."}:
        raise ValueError("Invalid preset_id.")
    if any(sep in normalized for sep in ("/", "\\", "\x00")):
        raise ValueError("Invalid preset_id.")
    return normalized


def _is_path_within(path: Path, base_dir: Path) -> bool:
    try:
        path.relative_to(base_dir)
        return True
    except ValueError:
        return False
