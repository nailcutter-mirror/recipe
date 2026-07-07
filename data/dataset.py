"""
Token-shard dataset for canonical training.

Each shard is a flat binary file of uint16 token ids (GPT-2 BPE fits in 16
bits). The dataset opens shards via memory-mapping for cheap random access,
and yields sequences of fixed length under a deterministic seed-keyed index
so the canonical training run can be re-executed bit-for-bit at the data
level (audit reproducibility).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch


def load_shard(path: Path | str) -> np.ndarray:
    """Memory-map a shard as uint16 array."""
    return np.memmap(path, dtype=np.uint16, mode="r")


class TokenShardDataset:
    """
    Deterministic packed-sequence iterator over a manifest's shards.

    The sequence at index i is a length-(seq_len+1) view starting at offset
    f(seed, i) inside the concatenated token stream. Targets are inputs
    shifted by one; we return tensors of length seq_len each.

    The deterministic data order is critical for audit reproducibility: a
    miner declares the seed, and the validator re-derives the exact same
    sequence of training examples on audit.
    """

    def __init__(
        self,
        manifest_path: Path | str,
        base_dir: Path | str,
        seq_len: int,
        seed: int,
    ):
        from .manifest import DataManifest, verify_manifest

        self.manifest = DataManifest.from_path(manifest_path)
        base = Path(base_dir)
        bad = verify_manifest(self.manifest, base)
        if bad:
            raise ValueError(f"manifest verification failed: {bad}")
        self._shards = [load_shard(base / s.relpath) for s in self.manifest.shards]
        self._cum = np.cumsum([0] + [len(s) for s in self._shards])
        self._total = int(self._cum[-1])
        self.seq_len = seq_len
        self.seed = seed
        if self._total < seq_len + 1:
            raise ValueError(f"not enough tokens ({self._total}) for seq_len {seq_len}")
        # No-replacement permuted-sweep bookkeeping. We tile the token stream into
        # `n_blocks` non-overlapping contiguous blocks of (seq_len+1) tokens each
        # and sweep them in a deterministic seed-keyed permutation per epoch. This
        # yields full-coverage, no-repeat-within-epoch sampling (a strictly better
        # gradient signal than with-replacement draws — no wasted duplicate windows
        # and no unseen holes) while remaining a pure deterministic function of
        # (seed, step): the validator re-derives the exact same order on audit.
        # Per-epoch permutations are MEMOIZED — never rebuilt per get_batch call —
        # so throughput is unaffected.
        self._n_blocks = max(1, self._total // (self.seq_len + 1))
        self._perm_cache: dict[int, np.ndarray] = {}

    @property
    def total_tokens(self) -> int:
        return self._total

    def __len__(self) -> int:
        # Effectively unbounded; we report the number of non-overlapping windows
        # for sizing purposes, but indexing wraps modulo total_tokens.
        return max(1, self._total // (self.seq_len + 1))

    def _global_token(self, global_idx: int) -> int:
        """Return token at global byte index."""
        # Locate shard.
        shard_idx = int(np.searchsorted(self._cum, global_idx, side="right") - 1)
        within = global_idx - int(self._cum[shard_idx])
        return int(self._shards[shard_idx][within])

    def _read_range(self, start: int, length: int) -> np.ndarray:
        """Read a contiguous range of `length` tokens starting at global `start`,
        wrapping around the total token stream as needed."""
        out = np.empty(length, dtype=np.uint16)
        filled = 0
        cursor = start % self._total
        while filled < length:
            shard_idx = int(np.searchsorted(self._cum, cursor, side="right") - 1)
            shard = self._shards[shard_idx]
            within = cursor - int(self._cum[shard_idx])
            take = min(length - filled, len(shard) - within, self._total - cursor)
            out[filled : filled + take] = shard[within : within + take]
            filled += take
            cursor = (cursor + take) % self._total
        return out

    def get(self, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (input_ids, target_ids) for training step `step`.

        Offsets are derived deterministically from (seed, step) so two runs
        with the same (manifest, seed, seq_len) consume the exact same
        sequence of training examples in the exact same order.
        """
        # Deterministic PRNG keyed by (seed, step).
        rng = np.random.default_rng(np.array([self.seed, step], dtype=np.uint64))
        start = int(rng.integers(0, self._total))
        chunk = self._read_range(start, self.seq_len + 1)
        ids = torch.from_numpy(chunk.astype(np.int64))
        return ids[:-1], ids[1:]

    def _epoch_perm(self, epoch: int) -> np.ndarray:
        """MEMOIZED deterministic permutation of the `n_blocks` contiguous blocks
        for the given epoch. Keyed by (seed, 0xE9C, epoch) so it is bit-identical
        across runs (validator re-derivation) and never rebuilt once cached."""
        perm = self._perm_cache.get(epoch)
        if perm is None:
            rng = np.random.default_rng(
                np.array([self.seed, 0xE9C, epoch], dtype=np.uint64)
            )
            perm = rng.permutation(self._n_blocks)
            self._perm_cache[epoch] = perm
        return perm

    def get_batch(
        self,
        step: int,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a batch of (B, T) input + target tensors at the given step.

        No-replacement permuted sweep: for row `b` the global block index is
        gi = step*batch_size + b, decomposed into epoch = gi // n_blocks and
        pos = gi % n_blocks; the sampled block is memoized_perm(epoch)[pos] and
        the window starts at block*(seq_len+1). Deterministic in (seed, step, b)
        and bit-reproducible; per-epoch perms are memoized so tok/s is unaffected.
        """
        n_blocks = self._n_blocks
        stride = self.seq_len + 1
        inputs = np.empty((batch_size, self.seq_len), dtype=np.int64)
        targets = np.empty((batch_size, self.seq_len), dtype=np.int64)
        for b in range(batch_size):
            gi = step * batch_size + b
            epoch = gi // n_blocks
            pos = gi % n_blocks
            block = int(self._epoch_perm(epoch)[pos])
            start = block * stride
            chunk = self._read_range(start, stride).astype(np.int64)
            inputs[b] = chunk[:-1]
            targets[b] = chunk[1:]
        return torch.from_numpy(inputs), torch.from_numpy(targets)
