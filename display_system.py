import threading
import queue
import time
import sys
from typing import List, Optional

class DisplaySystem:
    """Manages the console display of subtitles."""
    
    def __init__(self, input_queue: queue.Queue):
        self.input_queue = input_queue # Receives Subtitle dicts
        
        self.subtitle_history: List[dict] = []
        self.max_history = 5
        self.current_subtitle: Optional[dict] = None
        
        self._running = False
        self.thread = None
        
        self.terminal_width = 80
        
    def start(self):
        if self._running: return
        print("[Display] Starting...")
        self._running = True
        self.thread = threading.Thread(target=self._run, name="Display")
        self.thread.start()
    
    def stop(self):
        if not self._running: return
        print("[Display] Stopping...")
        self._running = False
        if self.thread:
            self.input_queue.put(None) # Sentinel
            self.thread.join()
        print("\n[Display] Stopped.")
    
    def _run(self):
        """Main display loop"""
        while self._running:
            try:
                subtitle = self.input_queue.get(timeout=0.1)
                
                if subtitle is None: # Sentinel
                    continue
                
                # Add to history
                if self.current_subtitle:
                    self.subtitle_history.append(self.current_subtitle)
                if len(self.subtitle_history) > self.max_history:
                    self.subtitle_history.pop(0)
                
                # Set current
                self.current_subtitle = subtitle
                
                self._render_console()
                
            except queue.Empty:
                # No new subtitles, just idle
                pass
            
            except Exception as e:
                print(f"[Display] Error: {e}")
    
    def _render_console(self):
        """Render to console/terminal"""
        # Clear screen
        print("\033[2J\033[H", end="")
        
        print("=" * self.terminal_width)
        print(" LIVE INDIC TRANSLATION ".center(self.terminal_width))
        print("=" * self.terminal_width)
        
        # History
        print("\n--- History ---")
        if self.subtitle_history:
            for sub in self.subtitle_history:
                print(f"  \033[2m{sub['translated_text'][:70]}\033[0m") # Dim
        else:
            print("  [No history yet]")
            
        print("\n--- Current ---")
        
        if self.current_subtitle:
            source = self.current_subtitle['source_text']
            translated = self.current_subtitle['translated_text']
            
            print(f"\n  SRC: {source}")
            
            # Highlighted translated text
            print(f"\n  \033[1m\033[92m{translated}\033[0m\n") # Bold Green
        else:
            print("\n  [Waiting for speech...]\n")
        
        print("=" * self.terminal_width)
        print("Press Ctrl+C to stop")
        sys.stdout.flush()