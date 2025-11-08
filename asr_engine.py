import sys
import os
import json
import torch
import numpy as np
import queue
import threading
import time
from ring_buffer import TimestampedRingBuffer

# Attempt import of the custom ONNX wrapper from the model dir
try:
    from model_onnx import IndicASRModel, IndicASRConfig
    IMPORTED_CUSTOM_ASR = True
except ImportError:
    IMPORTED_CUSTOM_ASR = False

class ASREngine:
    """
    Wraps the IndicConformer ONNX model for inference.
    Current mode: segment-level transcription (batch from ring).
    Prep for future streaming stabilization (token-level) by keeping API clean.
    """
    def __init__(self, model_path: str, lang: str = "hi"):
        self.model_path = os.path.abspath(model_path)
        self.lang = lang
        self.model = None
        self.sample_rate = 16000

        if not IMPORTED_CUSTOM_ASR:
            # Try path injection
            sys.path.insert(0, self.model_path)
            try:
                from model_onnx import IndicASRModel, IndicASRConfig  # noqa
                print("[ASREngine] Imported custom classes after path insert.")
            except ImportError as e:
                raise ImportError(f"Could not import IndicASRModel from {self.model_path}. Error: {e}")

        # Load config.json from the model path
        config_path = os.path.join(self.model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"ASR config.json not found at {config_path}")

        print(f"[ASREngine] Loading config: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        config = IndicASRConfig(**cfg)
        config.ts_folder = self.model_path

        print("[ASREngine] Initializing IndicASRModel...")
        self.model = IndicASRModel(config)
        print("[ASREngine] ASR model loaded.")

    def transcribe(self, audio_segment_np: np.ndarray) -> str:
        """
        Segment-level transcription. Input: float32 mono (-1..1).
        Returns a single finalized text string for the segment.
        """
        if self.model is None:
            return ""
        if audio_segment_np.ndim == 1:
            tensor = torch.from_numpy(audio_segment_np).float().unsqueeze(0)
        else:
            tensor = torch.from_numpy(audio_segment_np).float()

        try:
            with torch.no_grad():
                text = self.model(tensor, self.lang)
            return text or ""
        except Exception as e:
            print(f"[ASREngine] Transcription error: {e}")
            return ""

class ASRWorker:
    """
    Pulls finalized audio spans (by chunk_id range) from the ring,
    runs ASR once per span, and emits a single finalized segment.
    """
    def __init__(self,
                 asr_engine: ASREngine,
                 ring_buffer: TimestampedRingBuffer,
                 input_queue: queue.Queue,
                 output_queue: queue.Queue):
        self.asr_engine = asr_engine
        self.ring = ring_buffer
        self.input_queue = input_queue     # jobs: {start_id, end_id, start_time, end_time}
        self.output_queue = output_queue   # emits: {text, start_time, end_time, lang}
        self._running = False
        self.thread = None

    def start(self):
        if self._running: return
        print("[ASRWorker] Starting...")
        self._running = True
        self.thread = threading.Thread(target=self._run, name="ASRWorker", daemon=True)
        self.thread.start()

    def stop(self):
        if not self._running: return
        print("[ASRWorker] Stopping...")
        self._running = False
        if self.thread:
            self.input_queue.put(None)
            self.thread.join()
        print("[ASRWorker] Stopped.")

    def _run(self):
        while self._running:
            try:
                job = self.input_queue.get(timeout=0.2)
                if job is None:
                    continue

                start_id = job['start_id']
                end_id   = job['end_id']
                t0 = job['start_time']
                t1 = job['end_time']

                # print(f"[ASRWorker] Job: chunks {start_id}-{end_id}  ({(t1 - t0)*1000:.0f} ms)")

                audio = self.ring.get_chunks(list(range(start_id, end_id + 1)))
                if audio is None:
                    print(f"[ASRWorker] Missing audio for {start_id}-{end_id} (evicted?). Skipping.")
                    continue

                t_asr0 = time.time()
                text = self.asr_engine.transcribe(audio)
                asr_ms = (time.time() - t_asr0) * 1000.0

                if not text.strip():
                    # print(f"[ASRWorker] Empty ASR result for {start_id}-{end_id}")
                    continue

                # print(f"[ASRWorker] Text: '{text}'  | ASR {asr_ms:.0f} ms")

                segment = {
                    'text': text,
                    'start_time': t0,
                    'end_time': t1,
                    'lang': self.asr_engine.lang
                }
                self.output_queue.put(segment)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ASRWorker] Error: {e}")
                import traceback; traceback.print_exc()