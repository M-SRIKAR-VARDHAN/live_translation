"""
A fixed-size, thread-safe, timestamped ring buffer for audio chunks.

Key upgrades:
- Sample-clocked timestamps (t_start / t_end) from a monotonic sample counter.
- Stores raw int16 to minimize memory; exposes normalized float32 when read.
- Backward compatible: put() still returns chunk_id; get()/get_chunks() still work.
- Optional multi-reader support via register_reader() + read_available() (zero-copy views).
"""

from __future__ import annotations
import numpy as np
import threading
from collections import deque
from typing import Optional, List, Dict, Tuple

class TimestampedRingBuffer:
    def __init__(
        self,
        max_chunks: int = 200,        # ~20 s at 100 ms
        chunk_size: int = 1600,       # 100 ms at 16 kHz
        sample_rate: int = 16000
    ):
        self.max_chunks = max_chunks
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate

        # Deque of dicts: {id, start_sample, end_sample, t_start, t_end, data_i16(np.ndarray[int16])}
        self._buffer: deque = deque(maxlen=max_chunks)
        self._lock = threading.Lock()

        # Monotonic counters
        self._next_chunk_id = 0
        self._next_start_sample = 0  # absolute sample index for next chunk start

        # Optional multi-reader cursors: reader_id -> last_seen_chunk_id
        self._readers: Dict[str, int] = {}

    # ---------------------------
    # Write path
    # ---------------------------
    def put(self, audio_bytes: bytes) -> int:
        """
        Add raw PCM int16 audio bytes (exactly chunk_size samples) to the ring.
        Returns the assigned chunk_id (monotonic).
        """
        # Convert to int16 array (no normalization here; keep memory small & IO fast)
        data_i16 = np.frombuffer(audio_bytes, dtype=np.int16)
        if data_i16.size != self.chunk_size:
            # Be strict: the producer should deliver exact chunk_size frames
            raise ValueError(
                f"Expected {self.chunk_size} samples, got {data_i16.size}."
            )

        with self._lock:
            cid = self._next_chunk_id
            start_sample = self._next_start_sample
            end_sample = start_sample + self.chunk_size

            t_start = start_sample / float(self.sample_rate)
            t_end = end_sample / float(self.sample_rate)

            self._buffer.append({
                "id": cid,
                "start_sample": start_sample,
                "end_sample": end_sample,
                "t_start": t_start,
                "t_end": t_end,
                "data_i16": data_i16.copy()  # copy because input buffer is reused by PyAudio
            })

            # Advance counters
            self._next_chunk_id += 1
            self._next_start_sample = end_sample

            # NOTE: When deque overflows, the oldest chunks are dropped automatically.

            return cid

    # ---------------------------
    # Legacy read API (ID-based)
    # ---------------------------
    def get(self, chunk_id: int) -> Optional[Dict]:
        """
        Return a dict for a specific chunk_id.
        Provides normalized float32 under key 'data' for backward compatibility.
        """
        with self._lock:
            for ch in self._buffer:
                if ch["id"] == chunk_id:
                    # Build a read-friendly view without copying int16 twice
                    data_f32 = ch["data_i16"].astype(np.float32) / 32768.0
                    return {
                        "id": ch["id"],
                        "start_sample": ch["start_sample"],
                        "end_sample": ch["end_sample"],
                        "timestamp": ch["t_start"],            # keep legacy key
                        "t_start": ch["t_start"],
                        "t_end": ch["t_end"],
                        "data": data_f32,                      # float32 (-1..1)
                        "data_i16": ch["data_i16"],            # int16 raw (zero-copy view)
                    }
        return None

    def get_chunks(self, chunk_ids: List[int]) -> Optional[np.ndarray]:
        """
        Concatenate a list of chunk_ids into a single float32 waveform (-1..1).
        Returns None if any requested chunk is missing (e.g., evicted).
        """
        with self._lock:
            # Fast index (id -> data_i16)
            idx = {ch["id"]: ch["data_i16"] for ch in self._buffer}

        out = []
        for cid in chunk_ids:
            arr = idx.get(cid, None)
            if arr is None:
                # requested chunk was evicted or never existed
                print(f"[RingBuffer] Warning: Chunk ID {cid} not found.")
                return None
            out.append(arr)

        if not out:
            return None

        # Convert once to float32 and concat
        return (np.concatenate(out).astype(np.float32)) / 32768.0

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    # ---------------------------
    # Optional: multi-reader API
    # ---------------------------
    def register_reader(self, reader_id: str) -> None:
        """
        Register a logical reader that will track its own position by chunk_id.
        On registration, the reader will start from the next chunk written.
        """
        with self._lock:
            # Reader's last_seen = last chunk already present (or -1 if empty)
            last_seen = self._next_chunk_id - 1
            self._readers[reader_id] = last_seen

    def read_available(self, reader_id: str, max_chunks: int = 8) -> List[Dict]:
        """
        Return up to max_chunks newest chunks after the reader's last seen id.
        Each item contains metadata and zero-copy int16 view (under 'data_i16').
        Advances the reader cursor.
        """
        with self._lock:
            if reader_id not in self._readers:
                raise KeyError(f"Reader '{reader_id}' not registered.")

            last_seen = self._readers[reader_id]
            # Collect chunks with id > last_seen (in order)
            available = [ch for ch in self._buffer if ch["id"] > last_seen]
            take = available[:max_chunks]

            if take:
                self._readers[reader_id] = take[-1]["id"]

            # Prepare lightweight views
            out = []
            for ch in take:
                out.append({
                    "id": ch["id"],
                    "start_sample": ch["start_sample"],
                    "end_sample": ch["end_sample"],
                    "t_start": ch["t_start"],
                    "t_end": ch["t_end"],
                    "data_i16": ch["data_i16"],  # zero-copy int16
                })
            return out
