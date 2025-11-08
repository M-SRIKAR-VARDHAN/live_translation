import torch
import numpy as np
import queue
import threading
from collections import deque
from typing import Optional
from ring_buffer import TimestampedRingBuffer

class VADModel:
    """
    Silero VAD wrapper with stability improvements:
      - 3-frame median smoothing on speech probability
      - Hysteresis thresholds (start > 0.6, end < 0.4) to prevent flicker
      - 16 kHz, processes 512-sample windows internally
    """
    def __init__(self, model_path: str):
        import os
        self.model = None
        self.sample_rate = 16000
        self.win = 512  # Silero's default step
        self._buf = np.array([], dtype=np.float32)

        # smoothing + hysteresis
        self._prob_hist = deque(maxlen=3)
        self._smoothing_enabled = True
        self._speak_state = False
        self._last_prob = 0.0
        
        # ==================================================================
        # <<< --- TUNING FIX --- >>>
        # We are making the VAD more sensitive
        # ==================================================================
        self.start_th = 0.40  # Lowered from 0.60 (easier to START speech)
        self.end_th = 0.30    # Lowered from 0.40 (harder to END speech)
        # ==================================================================


        # load TorchScript (preferred); fallback to torch.hub
        try:
            vad_ts = os.path.join(model_path, "silero_vad.jit")
            if not os.path.exists(vad_ts):
                raise FileNotFoundError(vad_ts)
            print(f"[VADModel] Loading TorchScript: {vad_ts}")
            self.model = torch.jit.load(vad_ts).eval()
            print("[VADModel] Loaded.")
        except Exception as e:
            print(f"[VADModel] TorchScript load failed ({e}); trying torch.hub...")
            self.model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad", model="silero_vad",
                force_reload=False, onnx=False
            )
            self.model.eval()
            print("[VADModel] Loaded via torch.hub.")

    def _smooth(self, p: float) -> float:
        if not self._smoothing_enabled:
            return p
        self._prob_hist.append(p)
        # 3-frame median is robust to spikes
        return float(np.median(self._prob_hist)) if self._prob_hist else p

    def process_chunk(self, audio_chunk_f32: np.ndarray) -> Optional[str]:
        """
        Append a 100ms chunk (1600 samples @16kHz) and emit a stable VAD event:
          - 'SPEECH_START' / 'SPEECH' / 'SPEECH_END' / 'SILENCE'
        """
        # accumulate
        self._buf = np.concatenate([self._buf, audio_chunk_f32])

        emitted_event: Optional[str] = None
        while self._buf.size >= self.win:
            frame = self._buf[:self.win]
            self._buf = self._buf[self.win:]

            with torch.inference_mode():
                prob = float(self.model(torch.from_numpy(frame).unsqueeze(0), self.sample_rate).item())

            prob = self._smooth(prob)

            # ==================================================================
            # <<< --- DEBUGGING RE-ADDED --- >>>
            # Let's see the probabilities again with the new thresholds
            # ==================================================================
            print(f"[VAD DEBUG] prob: {prob:.2f} | state: {self._speak_state} (Start Th: {self.start_th})")
            # ==================================================================


            if not self._speak_state:
                # currently silent; look for strong speech onset
                if prob >= self.start_th:
                    self._speak_state = True
                    emitted_event = "SPEECH_START"
                else:
                    emitted_event = "SILENCE"
            else:
                # currently speaking; look for strong offset
                if prob <= self.end_th:
                    self._speak_state = False
                    emitted_event = "SPEECH_END"
                else:
                    emitted_event = "SPEECH"

            self._last_prob = prob

        return emitted_event

    def reset_states(self):
        self._buf = np.array([], dtype=np.float32)
        self._prob_hist.clear()
        self._speak_state = False
        self._last_prob = 0.0


class VADWorker:
    """
    Listens for chunk IDs, fetches audio & timestamps from the ring,
    runs smoothed VAD, and emits events with ring-based times.
    """
    def __init__(
        self,
        vad_model: VADModel,
        ring_buffer: TimestampedRingBuffer,
        input_queue: queue.Queue,    # chunk IDs from AudioHandler
        output_queue: queue.Queue    # VAD events -> SmartSegmenter
    ):
        self.vad_model = vad_model
        self.ring = ring_buffer
        self.in_q = input_queue
        self.out_q = output_queue
        self._running = False
        self.thread: Optional[threading.Thread] = None

    def start(self):
        if self._running: return
        print("[VADWorker] Starting...")
        self._running = True
        self.thread = threading.Thread(target=self._run, name="VADWorker", daemon=True)
        self.thread.start()

    def stop(self):
        if not self._running: return
        print("[VADWorker] Stopping...")
        self._running = False
        if self.thread:
            self.in_q.put(None)
            self.thread.join()
        self.vad_model.reset_states()
        print("[VADWorker] Stopped.")

    def _run(self):
        while self._running:
            try:
                cid = self.in_q.get(timeout=0.1)
                if cid is None:
                    continue

                ch = self.ring.get(cid)
                if not ch:
                    continue

                # ring returns float32 normalized audio under 'data' (100 ms)
                event = self.vad_model.process_chunk(ch["data"])
                if not event:
                    continue

                # use ring sample-clocked timestamps; event anchors at chunk end
                evt = {
                    "type": "VAD",
                    "result": event,
                    "chunk_id": cid,
                    "timestamp": ch["t_end"],   # more intuitive for pause measurement
                    "t_start": ch["t_start"],
                    "t_end": ch["t_end"],
                }
                self.out_q.put(evt)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[VADWorker] Error: {e}")