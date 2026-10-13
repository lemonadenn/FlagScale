# Modified for the HorizonKV inference artifact; see NOTICE.
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F


def _parse_int_set(raw: str):
    raw = (raw or "").strip().lower()
    if raw in ("", "all"):
        return None
    out = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        out.add(int(item))
    return out


class AttentionMapRecorder:
    _instance = None

    def __init__(self):
        self.enabled = os.getenv("ATTN_DUMP", "0") == "1"
        self.dump_types = {
            t.strip().lower() for t in os.getenv("ATTN_DUMP_TYPES", "self,cross").split(",") if t.strip()
        }
        self.dump_blocks = _parse_int_set(os.getenv("ATTN_DUMP_BLOCKS", "0"))
        self.dump_dir = Path(os.getenv("ATTN_DUMP_DIR", "debug_attn"))
        self.dump_head = int(os.getenv("ATTN_DUMP_HEAD", "0"))
        self.dump_heads = self._parse_heads(os.getenv("ATTN_DUMP_HEADS", ""))
        self.query_stride = max(1, int(os.getenv("ATTN_DUMP_QUERY_STRIDE", "16")))
        self.key_stride = max(1, int(os.getenv("ATTN_DUMP_KEY_STRIDE", "16")))
        self.token_pool = max(1, int(os.getenv("ATTN_DUMP_TOKEN_POOL", "1")))
        self.max_maps = max(1, int(os.getenv("ATTN_DUMP_MAX_MAPS", "100")))
        self.max_query_tokens = max(1, int(os.getenv("ATTN_DUMP_MAX_QUERY_TOKENS", "256")))
        self.max_key_tokens = max(1, int(os.getenv("ATTN_DUMP_MAX_KEY_TOKENS", "512")))
        # Same pooled q/k as used to build attn; default on when ATTN_DUMP=1 unless explicitly disabled.
        _save_qk = os.getenv("ATTN_DUMP_SAVE_QK_SCORES")
        if _save_qk is None:
            self.save_qk_scores = os.getenv("ATTN_DUMP", "0") == "1"
        else:
            self.save_qk_scores = _save_qk == "1"
        self._num_saved = 0
        self._counter = 0
        if self.enabled:
            self.dump_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def _parse_heads(raw: str):
        raw = (raw or "").strip()
        if raw == "":
            return None
        heads = []
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            heads.append(int(item))
        return sorted(set(heads)) if heads else None

    def _should_dump(self, attn_type: str, block_index: int | None):
        if not self.enabled or self._num_saved >= self.max_maps:
            return False
        if attn_type not in self.dump_types:
            return False
        if self.dump_blocks is not None and block_index is not None and block_index not in self.dump_blocks:
            return False
        return True

    @staticmethod
    def _cap_tokens(x: torch.Tensor, max_tokens: int):
        if x.shape[1] <= max_tokens:
            return x
        step = math.ceil(x.shape[1] / max_tokens)
        return x[:, ::step, :]

    @staticmethod
    def _pool_tokens_avg(x: torch.Tensor, factor: int):
        """
        Average-pool on token axis [B, T, D] with kernel=stride=factor.
        Returns pooled tensor and representative indices in original token space.
        """
        if factor <= 1 or x.shape[1] < factor:
            indices = torch.arange(x.shape[1], device=x.device, dtype=torch.long)
            return x, indices

        # F.avg_pool1d expects [N, C, L], so treat D as channels.
        pooled = F.avg_pool1d(
            x.transpose(1, 2),
            kernel_size=factor,
            stride=factor,
            ceil_mode=True,
            count_include_pad=False,
        ).transpose(1, 2)

        n_bins = int(pooled.shape[1])
        start = torch.arange(n_bins, device=x.device, dtype=torch.long) * factor
        end = torch.clamp(start + factor, max=x.shape[1])
        center = (start + end - 1) // 2
        return pooled, center

    def record(
        self,
        attn_type: str,
        q: torch.Tensor,
        k: torch.Tensor,
        block_index: int | None = None,
        timestep: int | None = None,
        current_start: int | None = None,
        tag: str = "",
    ):
        if not self._should_dump(attn_type, block_index):
            return
        if q.ndim != 4 or k.ndim != 4:
            return
        if q.shape[0] == 0 or k.shape[0] == 0:
            return

        with torch.no_grad():
            head_limit = min(q.shape[2], k.shape[2])
            target_heads = self.dump_heads if self.dump_heads is not None else [self.dump_head]
            target_heads = [h for h in target_heads if 0 <= h < head_limit]

            for head_index in target_heads:
                if self._num_saved >= self.max_maps:
                    break

                qh = q[0:1, :, head_index, :].float()
                kh = k[0:1, :, head_index, :].float()

                qh, q_indices = self._pool_tokens_avg(qh, self.token_pool)
                kh, k_indices = self._pool_tokens_avg(kh, self.token_pool)
                q_pos = torch.arange(qh.shape[1], device=qh.device)[::self.query_stride]
                k_pos = torch.arange(kh.shape[1], device=kh.device)[::self.key_stride]
                qh = qh[:, q_pos, :]
                kh = kh[:, k_pos, :]
                q_indices = q_indices[q_pos]
                k_indices = k_indices[k_pos]
                qh = self._cap_tokens(qh, self.max_query_tokens)
                kh = self._cap_tokens(kh, self.max_key_tokens)
                q_indices = q_indices[:qh.shape[1]]
                k_indices = k_indices[:kh.shape[1]]

                scores = torch.matmul(qh, kh.transpose(-1, -2)) / math.sqrt(qh.shape[-1])
                attn = torch.softmax(scores, dim=-1)[0].detach()

                layer_index = block_index if block_index is not None else -1
                step_value = timestep if timestep is not None else -1
                start_value = current_start if current_start is not None else -1
                name = (
                    f"{self._counter:06d}_"
                    f"{attn_type}_"
                    f"layer{layer_index}_"
                    f"head{head_index}_"
                    f"timestep{step_value}_"
                    f"start{start_value}"
                )
                if tag:
                    name = f"{name}_{tag}"
                name = name.replace("/", "_")

                record = {
                    "attn": attn.cpu(),
                    "meta": {
                        "attn_type": attn_type,
                        "layer": layer_index,
                        "head": head_index,
                        "timestep": step_value,
                        "start": start_value,
                        "token_pool": self.token_pool,
                        "query_stride": self.query_stride,
                        "key_stride": self.key_stride,
                        "query_tokens": int(attn.shape[0]),
                        "key_tokens": int(attn.shape[1]),
                        "query_tokens_before_cap": int(q.shape[1]),
                        "key_tokens_before_cap": int(k.shape[1]),
                        "has_qk": self.save_qk_scores,
                    },
                    "query_indices": q_indices.cpu(),
                    "key_indices": k_indices.cpu(),
                }

                if self.save_qk_scores:
                    # Pooled + strided q/k (same shapes as attn axes); scores = qk^T/sqrt(d) before softmax.
                    record["q"] = qh[0].cpu()
                    record["k"] = kh[0].cpu()
                    record["scores"] = scores[0].cpu()

                tensor_path = self.dump_dir / f"{name}.pt"
                torch.save(record, tensor_path)

                self._counter += 1
                self._num_saved += 1
