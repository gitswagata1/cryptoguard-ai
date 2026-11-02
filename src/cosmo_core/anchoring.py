"""Merkle tree (SHA3-256) and append-only hash-chained JSONL ledger."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any


def _sha3_256(data: bytes) -> bytes:
    return hashlib.sha3_256(data).digest()


def compute_merkle_root(leaves: list[bytes]) -> bytes:
    """Compute a SHA3-256 Merkle tree root from leaf hashes."""
    if not leaves:
        return _sha3_256(b"")

    layer = [_sha3_256(leaf) if len(leaf) != 32 else leaf for leaf in leaves]

    while len(layer) > 1:
        next_layer: list[bytes] = []
        for i in range(0, len(layer), 2):
            if i + 1 < len(layer):
                combined = layer[i] + layer[i + 1]
            else:
                combined = layer[i] + layer[i]  # duplicate odd leaf
            next_layer.append(_sha3_256(combined))
        layer = next_layer

    return layer[0]


class AppendOnlyLedger:
    """Hash-chained JSONL ledger with recovery and integrity verification."""

    GENESIS_HASH: bytes = b"\x00" * 32

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._prev_hash: bytes = self.GENESIS_HASH
        self._count: int = 0
        self._recover()

    def _recover(self) -> None:
        """Recover chain state from an existing ledger file."""
        if not self._path.exists():
            return
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                self._prev_hash = bytes.fromhex(record["hash"])
                self._count += 1

    @property
    def count(self) -> int:
        return self._count

    @property
    def prev_hash(self) -> bytes:
        return self._prev_hash

    def commit(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Append a hash-chained record and return it."""
        record = {
            "seq": self._count,
            "timestamp": time.time(),
            "prev_hash": self._prev_hash.hex(),
            "payload": payload,
        }
        record_bytes = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
        record_hash = _sha3_256(record_bytes)
        record["hash"] = record_hash.hex()

        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")

        self._prev_hash = record_hash
        self._count += 1
        return record

    def verify(self) -> bool:
        """Verify the integrity of the entire chain."""
        if not self._path.exists():
            return self._count == 0

        prev = self.GENESIS_HASH
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                stored_hash = record.pop("hash")
                if record["prev_hash"] != prev.hex():
                    return False
                record_bytes = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
                computed = _sha3_256(record_bytes)
                if computed.hex() != stored_hash:
                    return False
                prev = computed
        return True
