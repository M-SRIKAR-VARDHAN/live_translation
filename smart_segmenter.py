import threading
import queue
import time
from typing import Optional, Set

class SmartSegmenterWorker:
    """
    Decides when to COMMIT an audio span for ASR based on:
      - Stable VAD pauses (>= pause_threshold_ms)
      - A time-cap commit if speech drags on (max_segment_duration_s)
      - A short cooldown to merge micro-pauses
    NOTE: Linguistic completeness (conjunction/punct gating) will be added
          when ASR stabilization lands in the next pass (needs text).
    """
    def __init__(
        self,
        audio_input_queue: queue.Queue,   # chunk IDs from AudioHandler
        vad_input_queue: queue.Queue,     # VAD events from VADWorker
        asr_job_queue: queue.Queue        # jobs -> ASRWorker
    ):
        self.audio_q = audio_input_queue
        self.vad_q = vad_input_queue
        self.asr_q = asr_job_queue

        self._running = False
        self.thread: Optional[threading.Thread] = None

        # --- segmentation state ---
        self.is_speaking = False
        self.segment_start_id = -1
        self.segment_start_ts = 0.0    # seconds (sample-clocked from VAD)
        self.last_activity_cid = -1
        self.last_activity_ts = 0.0    # seconds (sample-clocked from VAD)
        self.pause_anchor_ts = None    # when we first observed silence
        self.cooldown_ms = 120         # merge tiny gaps

        # Tracks the *actual* current time from the ring, not just speech time
        self.current_ring_ts = 0.0

        # thresholds
        self.pause_threshold_ms = 700      # min stable pause to commit
        self.max_segment_duration_s = 10   # force cut if talking forever
        self.min_segment_duration_ms = 200 # drop too-short blips
        self.time_cap_after_pause_ms = 1500  # if not linguistically complete (next pass), commit anyway

        # conjunction guard (placeholder-only until text available)
        self.conjunctions_en: Set[str] = {"and", "but", "so", "then", "or"}
        self.conjunctions_hi: Set[str] = {"और", "लेकिन", "पर", "तो", "या"}  # future use with ASR text

    def start(self):
        if self._running: return
        print("[Segmenter] Starting...")
        self._running = True
        self.thread = threading.Thread(target=self._run, name="SmartSegmenter", daemon=True)
        self.thread.start()

    def stop(self):
        if not self._running: return
        print("[Segmenter] Stopping...")
        self._running = False
        if self.thread:
            self.vad_q.put(None)
            self.thread.join()
        print("[Segmenter] Stopped.")

    # --------------------------
    # Main loop
    # --------------------------
    def _run(self):
        last_ui = time.time()
        while self._running:
            try:
                # Drain VAD first (high priority for timing)
                drained = False
                while True:
                    try:
                        ev = self.vad_q.get_nowait()
                        if ev is None:
                            break
                        self._handle_vad_event(ev)
                        drained = True
                    except queue.Empty:
                        break

                # Drain a few audio chunk ids to keep track of last seen id
                try:
                    cid = self.audio_q.get_nowait()
                    self._handle_audio_chunk(cid)
                    drained = True
                except queue.Empty:
                    pass

                # Check timeouts (pause & max duration)
                self._check_timeouts()

                # occasional heartbeat
                if drained or (time.time() - last_ui) > 1.0:
                    last_ui = time.time()

                time.sleep(0.01)

            except Exception as e:
                print(f"[Segmenter] Error: {e}")
                self._reset_state()

    # --------------------------
    # Event handlers
    # --------------------------
    def _handle_audio_chunk(self, chunk_id: int):
        # only keep ID to compute [start_id .. last_id] span; timestamps come from VAD events
        if self.is_speaking:
            self.last_activity_cid = chunk_id

    def _handle_vad_event(self, ev: dict):
        result = ev["result"]
        ts_end = ev.get("t_end", ev.get("timestamp", 0.0))  # prefer end time for pause math
        cid = ev["chunk_id"]
        
        # Always update the "current time"
        self.current_ring_ts = ts_end

        if result == "SPEECH_START":
            # merge tiny gaps: if we were silent for < cooldown, keep same segment
            if not self.is_speaking and self.pause_anchor_ts is not None:
                gap_ms = (ts_end - self.pause_anchor_ts) * 1000.0
                if gap_ms <= self.cooldown_ms:
                    # treat as continuous speech
                    self.is_speaking = True
                    self.last_activity_cid = cid
                    self.last_activity_ts = ts_end
                    self.pause_anchor_ts = None
                    return

            # new segment
            self.is_speaking = True
            self.segment_start_id = cid
            self.segment_start_ts = ts_end
            self.last_activity_cid = cid
            self.last_activity_ts = ts_end
            self.pause_anchor_ts = None

        elif result == "SPEECH":
            if self.is_speaking:
                self.last_activity_cid = cid
                self.last_activity_ts = ts_end

        elif result == "SPEECH_END":
            # mark potential pause start; we won't commit immediately
            self.is_speaking = False
            self.last_activity_cid = cid
            self.last_activity_ts = ts_end
            self.pause_anchor_ts = ts_end

        else:  # "SILENCE" or unknown
            # nothing special; _check_timeouts will decide based on pause_anchor_ts
            pass

    # --------------------------
    # Timeout logic
    # --------------------------
    def _check_timeouts(self):
        # Use the *actual* current time, not the time of the last speech event
        now_ts = self.current_ring_ts

        # 1) Pause-based commit
        if not self.is_speaking and self.segment_start_id != -1 and self.pause_anchor_ts is not None:
            silence_ms = (now_ts - self.pause_anchor_ts) * 1000.0
            if silence_ms >= self.pause_threshold_ms:
                # We could add linguistic completeness checks here once ASR text is available.
                # For now, we also enforce a time-cap after pause to avoid stalling.
                self._commit_segment(reason="Pause Timeout")
                return

        # 2) Max duration cut while still speaking
        if self.is_speaking and self.segment_start_id != -1:
            dur_s = max(0.0, now_ts - self.segment_start_ts)
            if dur_s >= self.max_segment_duration_s:
                self._commit_segment(reason="Max Duration")
                return

        # 3) Safety: if we ended speech, but no commit after a long time, force commit
        if not self.is_speaking and self.segment_start_id != -1 and self.pause_anchor_ts is not None:
            since_end_ms = (now_ts - self.pause_anchor_ts) * 1000.0
            if since_end_ms >= self.time_cap_after_pause_ms:
                self._commit_segment(reason="Time-cap After Pause")
                return

    # --------------------------
    # Commit & reset
    # --------------------------
    def _commit_segment(self, reason: str):
        if self.segment_start_id == -1 or self.last_activity_cid == -1:
            self._reset_state()
            return

        duration_ms = max(0.0, (self.last_activity_ts - self.segment_start_ts) * 1000.0)
        if duration_ms < self.min_segment_duration_ms:
            # print(f"[Segmenter] Discard short segment: {duration_ms:.0f} ms ({reason})")
            self._reset_state()
            return

        start_id = self.segment_start_id
        end_id = self.last_activity_cid

        job = {
            "start_id": start_id,
            "end_id": end_id,
            "start_time": self.segment_start_ts,
            "end_time": self.last_activity_ts,
            "reason": reason,
        }
        # print(f"[Segmenter] COMMIT {start_id}->{end_id} ({duration_ms:.0f} ms) [{reason}]")
        self.asr_q.put(job)
        self._reset_state()

    def _reset_state(self):
        self.is_speaking = False
        self.segment_start_id = -1
        self.segment_start_ts = 0.0
        self.last_activity_cid = -1
        self.last_activity_ts = 0.0
        self.pause_anchor_ts = None