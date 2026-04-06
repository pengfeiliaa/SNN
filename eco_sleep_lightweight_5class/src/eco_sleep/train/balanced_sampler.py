"""批次均衡采样器：在每个 batch 内尽量覆盖所有类别，缓解长尾塌缩。"""

from __future__ import annotations

from typing import Iterator, List
import math

import numpy as np
from torch.utils.data import Sampler


class BalancedBatchSampler(Sampler[List[int]]):
    """按类别均衡构造 batch。

    每个 batch 尽量让各类样本数量相近；当某类样本不足时允许有放回采样，
    以确保少数类在每个 batch 中都出现，避免梯度完全被多数类淹没。
    """

    def __init__(
        self,
        labels: np.ndarray,
        num_classes: int,
        batch_size: int,
        drop_last: bool = True,
        seed: int = 42,
    ) -> None:
        labels = np.asarray(labels, dtype=np.int64)
        if labels.ndim != 1:
            raise ValueError("labels 必须是一维数组，用于构建均衡 batch。")
        self.num_classes = int(num_classes)
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        if self.num_classes <= 1:
            raise ValueError("num_classes 必须大于 1。")
        if self.batch_size < self.num_classes:
            raise ValueError("batch_size 不能小于 num_classes，否则无法保证每类都出现。")

        self.labels = labels
        self.num_samples = int(labels.size)
        self.rng = np.random.default_rng(seed)

        self.class_indices = [np.flatnonzero(labels == c) for c in range(self.num_classes)]
        for c, idx in enumerate(self.class_indices):
            if idx.size == 0:
                raise ValueError(f"第 {c} 类样本数为 0，无法构造均衡 batch。")

    def __len__(self) -> int:
        if self.batch_size <= 0:
            return 0
        if self.drop_last:
            return self.num_samples // self.batch_size
        return int(math.ceil(self.num_samples / self.batch_size))

    def __iter__(self) -> Iterator[List[int]]:
        num_batches = len(self)
        per_class = self.batch_size // self.num_classes
        remainder = self.batch_size - per_class * self.num_classes
        classes = np.arange(self.num_classes)

        for _ in range(num_batches):
            if remainder > 0:
                extra = self.rng.choice(classes, size=remainder, replace=False)
                extra_set = set(extra.tolist())
            else:
                extra_set = set()

            batch: List[int] = []
            for c in classes:
                k = per_class + (1 if c in extra_set else 0)
                if k <= 0:
                    continue
                idxs = self.class_indices[c]
                replace = idxs.size < k
                chosen = self.rng.choice(idxs, size=k, replace=replace)
                batch.extend(chosen.tolist())

            self.rng.shuffle(batch)
            yield batch
