"""
Context-Aware Translation Engine and Worker
Now with:
  - Source-side context cache (unchanged)
  - Diffed output (emit only NEW translated suffix)
  - Queue shedding (drop oldest jobs if backlog grows)
  - End-to-end latency logging (t_display - t_end_of_audio)
"""

from __future__ import annotations
import time
import threading
import queue
from typing import Optional, List
from dataclasses import dataclass
from collections import deque
import torch

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from IndicTransToolkit.processor import IndicProcessor
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("FATAL: transformers or IndicTransToolkit not available.")
    print("Please run: pip install transformers 'IndicTransToolkit @ git+https://github.com/VarunGumma/IndicTransToolkit.git'")

# ------------------
# Utility: diff tail
# ------------------
def _suffix_diff(prev: str, now: str) -> str:
    """
    Return ONLY the new suffix in 'now' that wasn't in 'prev'.
    Simple and robust: find longest common prefix, emit remainder.
    """
    if not prev:
        return now
    i = 0
    L = min(len(prev), len(now))
    while i < L and prev[i] == now[i]:
        i += 1
    return now[i:]

@dataclass
class TranslationResult:
    source_text: str
    translated_text: str       # full translation
    diff_text: str             # only new suffix for display
    source_lang: str
    target_lang: str
    start_time: float
    end_time: float
    t_infer_ms: float

# --- MODEL CLASS ---

class TranslationEngine:
    """
    Wraps IndicTrans2 with a source-language context cache.
    Adds diffing support by remembering the last FULL target string we emitted.
    """
    def __init__(self, model_path: str, context_size: int = 2):
        if not HAS_TRANSFORMERS:
            raise ImportError("TranslationEngine requires transformers and IndicTransToolkit")

        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.ip = None

        # Context: store last N source sentences
        self.context_cache = deque(maxlen=context_size)

        # For diffing: remember last FULL target we displayed
        self._last_target_full = ""

        print(f"[Translator] Loading model from: {self.model_path} on {self.device}")
        self._load_models()
        print("✓ [Translator] Model, Tokenizer, and IndicProcessor loaded.")

    def _load_models(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, local_files_only=True
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_path, trust_remote_code=True,
            low_cpu_mem_usage=True, local_files_only=True
        ).to(self.device)
        if self.device == "cuda":
            self.model.half()
        self.model.eval()
        self.ip = IndicProcessor(inference=True)

    def translate(self, source_text: str, source_lang: str, target_lang: str = "eng_Latn") -> Optional[str]:
        """
        Translate ONE new source segment with context.
        Returns the FULL translation string for the contextual input.
        """
        if not source_text or not source_text.strip():
            return None

        # Map ISO-ish to IndicTrans tags
        lang_map = {
            'hi': 'hin_Deva', 'te': 'tel_Telu', 'ta': 'tam_Taml',
            'bn': 'ben_Beng', 'kn': 'kan_Knda', 'ml': 'mal_Mlym',
            'gu': 'guj_Gujr', 'mr': 'mar_Deva', 'pa': 'pan_Guru',
            'as': 'asm_Beng', 'en': 'eng_Latn', 'und': 'eng_Latn',
        }
        mapped_source_lang = lang_map.get(source_lang, "hin_Deva")

        if mapped_source_lang == target_lang:
            # Identity: still update context (source)
            self.context_cache.append(source_text)
            return source_text

        # Stitch context (source-only)
        context = " ".join(self.context_cache)
        full_input_text = f"{context} {source_text}".strip()

        try:
            # Preprocess
            processed = self.ip.preprocess_batch(
                [full_input_text],
                src_lang=mapped_source_lang,
                tgt_lang=target_lang
            )

            # Tokenize
            inputs = self.tokenizer(
                processed, truncation=True, padding="longest",
                return_tensors="pt", return_attention_mask=True
            ).to(self.device)

            # Generate
            start = time.time()
            with torch.no_grad():
                gen = self.model.generate(
                    **inputs,
                    use_cache=False,          # CPU-friendly; set True on CUDA if stable
                    min_length=0,
                    max_length=512,
                    num_beams=1,             # greedy/beam=1 for low latency
                    do_sample=False
                )
            infer_ms = (time.time() - start) * 1000.0

            # Decode & postprocess
            decoded = self.tokenizer.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            outs = self.ip.postprocess_batch(decoded, lang=target_lang)
            final = outs[0] if outs else None

            # Update context (source-only)
            self.context_cache.append(source_text)

            if final is None:
                return None

            # Diff against last FULL target
            diff = _suffix_diff(self._last_target_full, final)
            # Update last FULL target
            self._last_target_full = final

            # Return both via packed str "final|||diff"? No—Worker constructs dict.
            return final, diff, infer_ms

        except Exception as e:
            print(f"[Translator] Error: {e}")
            import traceback; traceback.print_exc()
            return None

    def reset(self):
        self.context_cache.clear()
        self._last_target_full = ""


class TranslationWorker:
    """
    Runs the NMT engine in a separate thread.
    Adds queue shedding (drop oldest) when backlog grows.
    Emits both FULL translation and DIFF for UI.
    """
    def __init__(self, translation_engine: TranslationEngine,
                 input_queue: queue.Queue,
                 output_queue: queue.Queue,
                 max_backlog: int = 3):
        self.engine = translation_engine
        self.input_queue = input_queue    # Receives ASRSegment dicts
        self.output_queue = output_queue  # Emits Subtitle dicts
        self.max_backlog = max_backlog

        self._running = False
        self.thread = None

    def start(self):
        if self._running: return
        print("[Translator] Starting...")
        self._running = True
        self.thread = threading.Thread(target=self._run, name="Translator", daemon=True)
        self.thread.start()

    def stop(self):
        if not self._running: return
        print("[Translator] Stopping...")
        self._running = False
        if self.thread:
            self.input_queue.put(None)
            self.thread.join()
        print("[Translator] Stopped.")

    def _shed_if_needed(self):
        # If backlog exceeds limit, drop oldest jobs (preserve most recent segments)
        dropped = 0
        while self.input_queue.qsize() > self.max_backlog:
            try:
                old = self.input_queue.get_nowait()
                if old is None:
                    continue
                dropped += 1
            except queue.Empty:
                break
        if dropped:
            print(f"[Translator] Shedding: dropped {dropped} stale segment(s)")

    def _run(self):
        while self._running:
            try:
                self._shed_if_needed()
                segment = self.input_queue.get(timeout=0.2)
                if segment is None:
                    continue

                src_text = segment.get('text', '')
                src_lang = segment.get('lang', 'hi')
                t_start = segment.get('start_time', 0.0)
                t_end = segment.get('end_time', 0.0)

                print(f"[Translator] In: '{src_text}'")

                result = self.engine.translate(source_text=src_text, source_lang=src_lang, target_lang="eng_Latn")
                if result is None:
                    print("[Translator] Translation failed.")
                    continue

                full, diff, infer_ms = result
                # end-to-end latency: now - speech end
                e2e_ms = (time.time() - t_end) * 1000.0

                print(f"[Translator] Out(full): '{full}'  | diff: '{diff}'  | NMT {infer_ms:.0f} ms  | E2E {e2e_ms:.0f} ms")

                # Emit to display: show only DIFF (smooth streaming feel)
                subtitle = {
                    'source_text': src_text,
                    'translated_text': diff if diff.strip() else full,  # if no diff, send full once
                    'start_time': t_start,
                    'end_time': t_end,
                    'full_translation': full,
                    'nmt_ms': infer_ms,
                    'e2e_ms': e2e_ms,
                }
                self.output_queue.put(subtitle)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Translator] Worker Error: {e}")
