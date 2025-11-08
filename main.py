"""
Main Application: The Conductor

This script initializes all components (models and workers),
connects them with queues, and starts the full pipeline.
"""

import sys
import time
import threading
import queue
import signal
import os
import json

# Import all our components
from ring_buffer import TimestampedRingBuffer
from audio_handler import AudioHandler
from vad_component import VADModel, VADWorker
from asr_engine import ASREngine, ASRWorker
from translation_engine import TranslationEngine, TranslationWorker
from smart_segmenter import SmartSegmenterWorker
from display_system import DisplaySystem

class PipelineManager:
    """Manages the lifecycle of all components and threads."""
    
    def __init__(self, config):
        self.config = config
        self.workers = []
        self.audio_handler = None
        
        # --- 1. Create all communication queues ---
        print("[Pipeline] Creating communication queues...")
        self.q_audio_to_vad = queue.Queue(maxsize=10)
        self.q_audio_to_segmenter = queue.Queue(maxsize=10)
        self.q_vad_to_segmenter = queue.Queue(maxsize=10)
        self.q_segmenter_to_asr = queue.Queue(maxsize=5)
        self.q_asr_to_translator = queue.Queue(maxsize=5)
        self.q_translator_to_display = queue.Queue(maxsize=5)
        print("[Pipeline] Queues created.")

        # --- 2. Initialize Components ---
        print("\n[Pipeline] Initializing components...")
        
        # Ring Buffer (The Heart)
        self.ring_buffer = TimestampedRingBuffer()
        
        # Audio (The Producer)
        self.audio_handler = AudioHandler(self.ring_buffer)
        self.audio_handler.register_listener(self.q_audio_to_vad)
        self.audio_handler.register_listener(self.q_audio_to_segmenter)
        
        # VAD Model
        self.vad_model = VADModel(config['vad_model_path'])
        
        # ASR Model
        self.asr_engine = ASREngine(
            model_path=config['asr_model_path'],
            lang=config['source_language']
        )
        
        # Translation Model
        self.translation_engine = TranslationEngine(
            model_path=config['nmt_model_path']
        )
        print("[Pipeline] All models loaded.")

        # --- 3. Initialize Workers (Threads) ---
        print("\n[Pipeline] Initializing workers...")
        
        self.vad_worker = VADWorker(
            self.vad_model, self.ring_buffer, self.q_audio_to_vad, self.q_vad_to_segmenter
        )
        self.workers.append(self.vad_worker)
        
        self.segmenter_worker = SmartSegmenterWorker(
            self.q_audio_to_segmenter, self.q_vad_to_segmenter, self.q_segmenter_to_asr
        )
        self.workers.append(self.segmenter_worker)
        
        self.asr_worker = ASRWorker(
            self.asr_engine, self.ring_buffer, self.q_segmenter_to_asr, self.q_asr_to_translator
        )
        self.workers.append(self.asr_worker)
        
        self.translator_worker = TranslationWorker(
            self.translation_engine, self.q_asr_to_translator, self.q_translator_to_display
        )
        self.workers.append(self.translator_worker)
        
        self.display_system = DisplaySystem(self.q_translator_to_display)
        # Note: DisplaySystem is also a worker, added separately
        
        print("[Pipeline] All workers initialized.")
        print("\n✅ System ready!\n")
    
    def start(self):
        """Start all components in the correct order."""
        print("🚀 [Pipeline] Starting system...")
        
        # 1. Start Display (listens for final results)
        self.display_system.start()
        
        # 2. Start all worker threads
        for worker in self.workers:
            worker.start()
            
        # 3. Start audio capture (starts feeding the system)
        self.audio_handler.start()
        
        print("\n" + "="*70)
        print(" SYSTEM IS LIVE ".center(70))
        print("="*70)
        print(f"\n📢 Speak in {self.config['source_language']}... Press Ctrl+C to stop\n")
    
    def stop(self):
        """Stop all components in reverse order for clean shutdown."""
        print("\n\n🛑 [Pipeline] Stopping system...")
        
        # 1. Stop audio capture
        if self.audio_handler:
            self.audio_handler.stop()
        
        # 2. Stop all workers
        print("[Pipeline] Stopping workers...")
        for worker in self.workers:
            worker.stop()
        
        # 3. Stop display
        self.display_system.stop()
        
        # 4. Terminate PyAudio
        if self.audio_handler:
            self.audio_handler.terminate()
        
        print("[Pipeline] Shutdown complete. Goodbye!\n")

def main():
    """Entry point"""
    
    # --- Load Configuration ---
    config_path = 'model_paths.json'
    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        print("Please create it first (see documentation).")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # --- Run System ---
    manager = None
    try:
        manager = PipelineManager(config)
        
        # Handle Ctrl+C
        signal.signal(signal.SIGINT, lambda s, f: manager.stop())
        
        manager.start()
        
        # Keep main thread alive
        while manager and manager.audio_handler and manager.audio_handler._running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"❌ A fatal error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if manager:
            manager.stop()
        # Explicitly exit to ensure all daemon threads are killed
        sys.exit(0)


if __name__ == "__main__":
    main()