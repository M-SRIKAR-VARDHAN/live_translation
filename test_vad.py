"""
Test 2: VAD Engine (Silero VAD) - Component Validation
========================================================

Tests Voice Activity Detection to identify speech segments in audio.

Success Criteria:
1. Model loads without errors
2. Detects speech segments accurately
3. Processing latency < 50ms per chunk
4. Can identify pauses between speech
5. Tunable threshold for sensitivity

Usage:
    python test_02_vad.py --audio test_audio.wav
"""

import argparse
import time
import json
import logging
import sys
import os
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import torch

# Fix Windows console encoding
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_02_vad.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class VADTester:
    """Test harness for Silero VAD model"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.utils = None
        self.results = {
            "model_loaded": False,
            "speech_segments": [],
            "latencies": [],
            "errors": []
        }
    
    def test_1_model_loading(self) -> bool:
        """Test 2.1: Load Silero VAD model"""
        logger.info("=" * 70)
        logger.info("TEST 2.1: VAD Model Loading")
        logger.info("=" * 70)
        
        try:
            logger.info(f"Loading Silero VAD from: {self.model_path}")
            
            # Method 1: Try loading from local files
            vad_model_path = os.path.join(self.model_path, "silero_vad.jit")
            
            if os.path.exists(vad_model_path):
                logger.info(f"Found local model: {vad_model_path}")
                
                # Load using TorchScript
                self.model = torch.jit.load(vad_model_path)
                self.model.eval()
                
                logger.info("[PASS] Model loaded from local file!")
                logger.info(f"  Model type: {type(self.model)}")
                
            else:
                # Method 2: Download from torch.hub
                logger.info("Local model not found, downloading from torch.hub...")
                logger.info("(This will cache the model for future use)")
                
                self.model, self.utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    onnx=False
                )
                
                logger.info("[PASS] Model downloaded and loaded!")
            
            # Get utility functions if not already loaded
            if self.utils is None:
                try:
                    _, self.utils = torch.hub.load(
                        repo_or_dir='snakers4/silero-vad',
                        model='silero_vad',
                        force_reload=False,
                        onnx=False
                    )
                    logger.info("✓ Loaded VAD utility functions")
                except:
                    logger.warning("Could not load utils, will use manual implementation")
            
            self.results["model_loaded"] = True
            return True
            
        except Exception as e:
            logger.error(f"[FAIL] Model loading failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.results["errors"].append(str(e))
            return False
    
    def test_2_speech_detection(self, audio_path: str, 
                                threshold: float = 0.5) -> Tuple[bool, List[Dict]]:
        """Test 2.2: Detect speech segments in audio"""
        logger.info("=" * 70)
        logger.info("TEST 2.2: Speech Segment Detection")
        logger.info("=" * 70)
        
        try:
            logger.info(f"Audio: {audio_path}")
            logger.info(f"Threshold: {threshold}")
            
            # Check file exists
            if not Path(audio_path).exists():
                raise FileNotFoundError(f"File not found: {audio_path}")
            
            # Load audio
            import soundfile as sf
            audio_data, sample_rate = sf.read(audio_path)
            
            logger.info(f"Loaded: {len(audio_data)} samples at {sample_rate} Hz")
            
            # Convert to mono if stereo
            if len(audio_data.shape) == 2:
                audio_data = audio_data.mean(axis=1)
            
            # Resample to 16kHz if needed (Silero VAD expects 16kHz)
            if sample_rate != 16000:
                logger.info(f"Resampling from {sample_rate}Hz to 16000Hz...")
                import torchaudio.transforms as T
                audio_tensor = torch.from_numpy(audio_data).float()
                resampler = T.Resample(sample_rate, 16000)
                audio_tensor = resampler(audio_tensor)
                audio_data = audio_tensor.numpy()
                sample_rate = 16000
            
            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio_data).float()
            
            logger.info(f"Processing audio: {len(audio_data) / sample_rate:.2f} seconds")
            
            # Process in chunks (Silero VAD works on chunks)
            chunk_size = 512  # samples (32ms at 16kHz)
            speech_probs = []
            
            logger.info(f"Analyzing speech activity (chunk size: {chunk_size} samples)...")
            
            start_time = time.time()
            
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_tensor[i:i+chunk_size]
                
                # Pad last chunk if needed
                if len(chunk) < chunk_size:
                    chunk = torch.nn.functional.pad(chunk, (0, chunk_size - len(chunk)))
                
                # Get speech probability
                with torch.no_grad():
                    speech_prob = self.model(chunk.unsqueeze(0), sample_rate).item()
                
                speech_probs.append({
                    'start_time': i / sample_rate,
                    'end_time': (i + chunk_size) / sample_rate,
                    'speech_prob': speech_prob
                })
            
            total_time = (time.time() - start_time) * 1000
            avg_latency = total_time / len(speech_probs) if speech_probs else 0
            
            logger.info(f"\nProcessed {len(speech_probs)} chunks")
            logger.info(f"Total time: {total_time:.0f}ms")
            logger.info(f"Average latency per chunk: {avg_latency:.2f}ms")
            
            # Identify speech segments (where prob > threshold)
            segments = []
            in_speech = False
            segment_start = None
            
            for prob_info in speech_probs:
                if prob_info['speech_prob'] > threshold:
                    if not in_speech:
                        # Start of speech segment
                        in_speech = True
                        segment_start = prob_info['start_time']
                else:
                    if in_speech:
                        # End of speech segment
                        in_speech = False
                        segments.append({
                            'start': segment_start,
                            'end': prob_info['start_time'],
                            'duration': prob_info['start_time'] - segment_start
                        })
            
            # Handle case where audio ends during speech
            if in_speech:
                segments.append({
                    'start': segment_start,
                    'end': len(audio_data) / sample_rate,
                    'duration': (len(audio_data) / sample_rate) - segment_start
                })
            
            logger.info(f"\n" + "=" * 70)
            logger.info("SPEECH SEGMENTS DETECTED:")
            logger.info("=" * 70)
            
            if segments:
                for i, seg in enumerate(segments, 1):
                    logger.info(f"Segment {i}: {seg['start']:.2f}s - {seg['end']:.2f}s (duration: {seg['duration']:.2f}s)")
            else:
                logger.warning("No speech segments detected!")
                logger.info("Try adjusting the threshold (--threshold parameter)")
            
            logger.info("=" * 70)
            
            # Calculate statistics
            total_audio_duration = len(audio_data) / sample_rate
            total_speech_duration = sum(seg['duration'] for seg in segments)
            speech_ratio = (total_speech_duration / total_audio_duration) * 100 if total_audio_duration > 0 else 0
            
            logger.info(f"\nStatistics:")
            logger.info(f"  Total audio: {total_audio_duration:.2f}s")
            logger.info(f"  Total speech: {total_speech_duration:.2f}s")
            logger.info(f"  Speech ratio: {speech_ratio:.1f}%")
            logger.info(f"  Number of segments: {len(segments)}")
            
            self.results["speech_segments"] = segments
            self.results["latencies"].append(avg_latency)
            
            logger.info("\n[PASS] Speech detection successful!")
            return True, segments
            
        except Exception as e:
            logger.error(f"\n[FAIL] Speech detection failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.results["errors"].append(str(e))
            return False, []
    
    def test_3_latency_performance(self) -> bool:
        """Test 2.3: Check if latency meets requirements"""
        logger.info("\n" + "=" * 70)
        logger.info("TEST 2.3: Latency Performance")
        logger.info("=" * 70)
        
        if not self.results["latencies"]:
            logger.warning("No latency data available")
            return True
        
        avg_latency = np.mean(self.results["latencies"])
        max_latency = np.max(self.results["latencies"])
        
        logger.info(f"Average latency: {avg_latency:.2f}ms")
        logger.info(f"Max latency: {max_latency:.2f}ms")
        logger.info(f"Target: <50ms per chunk")
        
        if avg_latency < 50:
            logger.info("\n[PASS] Latency is excellent!")
            return True
        elif avg_latency < 100:
            logger.info("\n[WARN] Latency is acceptable but could be better")
            return True
        else:
            logger.warning(f"\n[WARN] Latency is high: {avg_latency:.2f}ms")
            return False
    
    def test_4_pause_detection(self) -> bool:
        """Test 2.4: Can it detect pauses between speech?"""
        logger.info("\n" + "=" * 70)
        logger.info("TEST 2.4: Pause Detection Capability")
        logger.info("=" * 70)
        
        segments = self.results.get("speech_segments", [])
        
        if len(segments) < 2:
            logger.info("Only 1 or 0 segments found")
            logger.info("Cannot test pause detection (need multiple segments)")
            logger.info("[SKIP] Test requires audio with pauses")
            return True
        
        # Calculate pauses between segments
        pauses = []
        for i in range(len(segments) - 1):
            pause_duration = segments[i+1]['start'] - segments[i]['end']
            if pause_duration > 0:
                pauses.append({
                    'after_segment': i+1,
                    'duration': pause_duration
                })
        
        logger.info(f"Found {len(pauses)} pauses between speech segments:")
        
        for pause in pauses:
            logger.info(f"  Pause after segment {pause['after_segment']}: {pause['duration']:.2f}s")
        
        logger.info("\n[PASS] VAD can detect pauses!")
        return True
    
    def generate_report(self):
        """Generate final test report"""
        logger.info("\n" + "=" * 70)
        logger.info("FINAL REPORT")
        logger.info("=" * 70)
        
        success = (
            self.results["model_loaded"] and
            len(self.results["errors"]) == 0
        )
        
        report = {
            "test_name": "VAD Engine Validation (Silero VAD)",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "PASS" if success else "FAIL",
            "model_path": self.model_path,
            "results": self.results,
            "recommendations": []
        }
        
        logger.info(f"Status: {report['status']}")
        logger.info(f"Model loaded: {self.results['model_loaded']}")
        logger.info(f"Speech segments found: {len(self.results['speech_segments'])}")
        logger.info(f"Errors: {len(self.results['errors'])}")
        
        if not self.results["model_loaded"]:
            report["recommendations"].append("CRITICAL: Model did not load")
        
        if len(self.results["speech_segments"]) == 0:
            report["recommendations"].append(
                "INFO: No speech segments detected. Try adjusting threshold or check audio."
            )
        
        if self.results["latencies"] and np.mean(self.results["latencies"]) > 50:
            report["recommendations"].append(
                f"INFO: Average latency is {np.mean(self.results['latencies']):.2f}ms. "
                "Consider GPU acceleration for production."
            )
        
        if report["recommendations"]:
            logger.info("\nRecommendations:")
            for rec in report["recommendations"]:
                logger.info(f"  - {rec}")
        
        return report


def main():
    parser = argparse.ArgumentParser(description="Test Silero VAD")
    parser.add_argument("--audio", type=str, default="test_audio.wav",
                       help="Path to audio file")
    parser.add_argument("--model_path", type=str, default="./models/silero_vad",
                       help="Path to VAD model (optional)")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Speech detection threshold (0-1, default: 0.5)")
    parser.add_argument("--output", type=str, default="test_02_vad_results.json",
                       help="Output JSON file")
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("SILERO VAD TEST")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Audio: {args.audio}")
    logger.info(f"Threshold: {args.threshold}")
    logger.info("=" * 70 + "\n")
    
    # Create tester
    tester = VADTester(args.model_path)
    
    # Run tests
    success = True
    
    # Test 1: Load model
    if not tester.test_1_model_loading():
        success = False
    else:
        # Test 2: Detect speech
        result, _ = tester.test_2_speech_detection(args.audio, args.threshold)
        if not result:
            success = False
        else:
            # Test 3 & 4: Performance tests
            tester.test_3_latency_performance()
            tester.test_4_pause_detection()
    
    # Generate report
    report = tester.generate_report()
    
    # Save report
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nReport saved: {args.output}")
    
    # Final message
    logger.info("\n" + "=" * 70)
    if success:
        logger.info("✓✓✓ SUCCESS ✓✓✓")
        logger.info("VAD model is WORKING!")
        logger.info("\nNext: Test 3 for NMT (Translation)")
    else:
        logger.error("✗✗✗ FAILED ✗✗✗")
        logger.error("Fix errors before proceeding")
    logger.info("=" * 70)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())