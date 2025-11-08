import pyaudio
import threading
import queue
import numpy as np
from typing import List, Optional
from ring_buffer import TimestampedRingBuffer

class AudioHandler:
    """
    Captures microphone audio at 16 kHz / mono / int16 and feeds the ring buffer.
    Fans out new chunk IDs to registered listener queues (non-blocking, drop-oldest on backpressure).

    Upgrades:
    - Strict chunk sizing (exactly N samples per ring buffer write).
    - Avoids intermediate conversions: PyAudio -> int16 bytes -> ring buffer.
    - Clear lifecycle (start/stop/terminate).
    """

    def __init__(
        self,
        ring_buffer: TimestampedRingBuffer,
        chunk_duration_ms: int = 100,
        rate: int = 16000,
        input_device_index: Optional[int] = None,
        channels: int = 1,
    ):
        self.ring_buffer = ring_buffer
        self.RATE = rate
        self.CHANNELS = channels

        # Calculate exact samples per chunk (must match ring chunk_size)
        self.CHUNK_SAMPLES = int(rate * (chunk_duration_ms / 1000.0))
        if self.CHUNK_SAMPLES != self.ring_buffer.chunk_size:
            # Keep the hot path consistent
            raise ValueError(
                f"AudioHandler chunk size ({self.CHUNK_SAMPLES}) must equal ring_buffer.chunk_size ({self.ring_buffer.chunk_size})."
            )

        self.FORMAT = pyaudio.paInt16
        self._input_device_index = input_device_index

        # Listener queues (receive chunk_id fan-out)
        self._listener_queues: List[queue.Queue] = []
        self._listener_lock = threading.Lock()

        self._p = pyaudio.PyAudio()
        self._stream: Optional[pyaudio.Stream] = None
        self._running = False

    # ---------------------------
    # Listener management
    # ---------------------------
    def register_listener(self, q: queue.Queue):
        """Registers a queue that will receive new chunk IDs."""
        with self._listener_lock:
            self._listener_queues.append(q)
        print(f"[AudioHandler] Registered new listener. Total: {len(self._listener_queues)}")

    # ---------------------------
    # PyAudio callback
    # ---------------------------
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """
        PyAudio supplies 'in_data' as bytes (int16 frames). We pass it straight into the ring buffer.
        The ring converts bytes -> np.int16 internally (and computes sample-clocked timestamps).
        """
        if not self._running:
            return (None, pyaudio.paComplete)

        try:
            # Sanity: ensure PyAudio gave us exactly CHUNK_SAMPLES frames
            # frame_count is #frames (samples per channel)
            if frame_count != self.CHUNK_SAMPLES:
                # Rare but can happen depending on host API. Accumulate or drop; here we drop to keep timing strict.
                # Alternatively, you could buffer leftovers; but strict 100 ms cadence is better for VAD/ASR sync.
                return (None, pyaudio.paContinue)

            # 1) Write to the ring (single source of truth)
            chunk_id = self.ring_buffer.put(in_data)

            # 2) Fan-out chunk_id to listeners (non-blocking; drop-oldest if full)
            with self._listener_lock:
                for q in self._listener_queues:
                    try:
                        q.put_nowait(chunk_id)
                    except queue.Full:
                        try:
                            _ = q.get_nowait()  # drop oldest
                            q.put_nowait(chunk_id)
                        except queue.Empty:
                            pass

            return (None, pyaudio.paContinue)

        except Exception as e:
            print(f"[AudioHandler] Callback error: {e}")
            return (None, pyaudio.paContinue)

    # ---------------------------
    # Lifecycle
    # ---------------------------
    def start(self):
        if self._running:
            return
        print("[AudioHandler] Starting audio stream...")

        self._running = True
        self._stream = self._p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK_SAMPLES,
            stream_callback=self._audio_callback,
            input_device_index=self._input_device_index
        )
        self._stream.start_stream()
        print(f"[AudioHandler] Audio stream started at {self.RATE} Hz, {self.CHANNELS} ch, {self.CHUNK_SAMPLES} samples per chunk.")

    def stop(self):
        if not self._running:
            return
        print("[AudioHandler] Stopping audio stream...")
        self._running = False

        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            finally:
                self._stream = None

        # Clear listener queues
        with self._listener_lock:
            for q in self._listener_queues:
                while not q.empty():
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        break

        print("[AudioHandler] Audio stream stopped.")

    def terminate(self):
        """Terminate PyAudio."""
        if self._p is not None:
            try:
                self._p.terminate()
            finally:
                self._p = None
            print("[AudioHandler] PyAudio terminated.")
