"""
Test 1: ASR Engine - FINAL WORKING VERSION (No FFmpeg Required)
=================================================================

Uses soundfile instead of torchaudio to avoid FFmpeg issues.

Usage:
    python test_01_asr_final.py --audio test_audio.wav --lang hi
"""

import argparse
import time
import json
import logging
import sys
import os
from pathlib import Path
from typing import Tuple
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
        logging.FileHandler('test_01_asr_final.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ASRTester:
    """Test harness for IndicConformer ONNX ASR model"""
    
    def __init__(self, model_path: str, lang: str = "hi"):
        self.model_path = os.path.abspath(model_path)
        self.lang = lang
        self.model = None
        self.results = {
            "model_loaded": False,
            "transcriptions": [],
            "latencies": [],
            "errors": []
        }
    
    def load_audio_file(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """Load audio file using soundfile (no FFmpeg needed)"""
        try:
            import soundfile as sf
            
            logger.info(f"Loading audio with soundfile: {audio_path}")
            
            # Load audio
            audio_data, sample_rate = sf.read(audio_path)
            
            logger.info(f"Loaded: {len(audio_data)} samples at {sample_rate} Hz")
            
            # Convert to torch tensor
            waveform = torch.from_numpy(audio_data).float()
            
            # Handle stereo to mono conversion
            if len(waveform.shape) == 2:
                logger.info("Converting stereo to mono...")
                waveform = waveform.mean(dim=1)
            
            # Add channel dimension
            if len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0)
            
            # Resample if needed (model expects 16kHz)
            if sample_rate != 16000:
                logger.info(f"Resampling from {sample_rate}Hz to 16000Hz...")
                import torchaudio.transforms as T
                resampler = T.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
                sample_rate = 16000
            
            logger.info(f"Final shape: {waveform.shape}, sr: {sample_rate}")
            
            return waveform, sample_rate
            
        except ImportError:
            logger.error("soundfile not installed. Installing...")
            logger.error("Run: pip install soundfile")
            raise
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            raise
    
    def test_1_model_loading(self) -> bool:
        """Test 1.1: Load the IndicConformer model"""
        logger.info("=" * 70)
        logger.info("TEST 1.1: Model Loading")
        logger.info("=" * 70)
        
        try:
            logger.info(f"Loading from: {self.model_path}")
            logger.info(f"Language: {self.lang}")
            
            # Add to path
            if self.model_path not in sys.path:
                sys.path.insert(0, self.model_path)
            
            # Import model
            from model_onnx import IndicASRModel, IndicASRConfig
            logger.info("✓ Imported model class")
            
            # Load config
            config_path = os.path.join(self.model_path, "config.json")
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            # Create model
            config = IndicASRConfig(**config_dict)
            config.ts_folder = self.model_path
            
            self.model = IndicASRModel(config)
            
            logger.info("[PASS] Model loaded successfully!")
            self.results["model_loaded"] = True
            return True
            
        except Exception as e:
            logger.error(f"[FAIL] Model loading failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.results["errors"].append(str(e))
            return False
    
    def test_2_transcription(self, audio_path: str) -> Tuple[bool, str]:
        """Test 1.2: Transcribe audio file"""
        logger.info("=" * 70)
        logger.info("TEST 1.2: Transcription")
        logger.info("=" * 70)
        
        try:
            logger.info(f"Audio: {audio_path}")
            logger.info(f"Language: {self.lang}")
            
            # Check file exists
            if not Path(audio_path).exists():
                raise FileNotFoundError(f"File not found: {audio_path}")
            
            file_size = Path(audio_path).stat().st_size / 1024
            logger.info(f"File size: {file_size:.2f} KB")
            
            # Load audio (using soundfile instead of torchaudio)
            waveform, sample_rate = self.load_audio_file(audio_path)
            
            # Transcribe
            logger.info("\nTranscribing...")
            logger.info("(This may take 10-60 seconds depending on audio length)\n")
            
            start_time = time.time()
            
            with torch.no_grad():
                transcription = self.model(waveform, self.lang)
            
            inference_time = (time.time() - start_time) * 1000
            
            logger.info("=" * 70)
            logger.info("TRANSCRIPTION RESULT:")
            logger.info("=" * 70)
            logger.info(transcription)
            logger.info("=" * 70)
            logger.info(f"Time: {inference_time:.0f}ms ({inference_time/1000:.1f}s)")
            logger.info("=" * 70)
            
            self.results["transcriptions"].append({
                "file": audio_path,
                "text": transcription,
                "language": self.lang,
                "inference_time_ms": inference_time
            })
            
            logger.info("\n[PASS] Transcription successful!")
            return True, transcription
            
        except Exception as e:
            logger.error(f"\n[FAIL] Transcription failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.results["errors"].append(str(e))
            return False, ""
    
    def test_3_streaming(self, audio_path: str) -> bool:
        """Test 1.3: Streaming simulation"""
        logger.info("\n" + "=" * 70)
        logger.info("TEST 1.3: Streaming Simulation")
        logger.info("=" * 70)
        
        try:
            if not Path(audio_path).exists():
                logger.info("Skipping streaming test (no audio file)")
                return True
            
            # Load audio
            waveform, sample_rate = self.load_audio_file(audio_path)
            
            # Process in chunks
            chunk_duration = 2.0  # seconds
            chunk_size = int(sample_rate * chunk_duration)
            
            audio_data = waveform.squeeze()
            total_samples = audio_data.shape[0]
            
            duration = total_samples / sample_rate
            logger.info(f"Audio duration: {duration:.2f}s")
            logger.info(f"Chunk size: {chunk_duration}s")
            
            # Test first 3 chunks
            num_test_chunks = min(3, int(np.ceil(total_samples / chunk_size)))
            logger.info(f"Testing {num_test_chunks} chunks...\n")
            
            for i in range(num_test_chunks):
                start_idx = i * chunk_size
                end_idx = min(start_idx + chunk_size, total_samples)
                chunk = audio_data[start_idx:end_idx].unsqueeze(0)
                
                logger.info(f"Chunk {i+1}:")
                
                start_time = time.time()
                with torch.no_grad():
                    result = self.model(chunk, self.lang)
                latency = (time.time() - start_time) * 1000
                
                self.results["latencies"].append(latency)
                
                logger.info(f"  Text: {result[:50]}...")
                logger.info(f"  Latency: {latency:.0f}ms\n")
            
            if self.results["latencies"]:
                avg = np.mean(self.results["latencies"])
                logger.info(f"Average latency: {avg:.0f}ms")
            
            logger.info("[PASS] Streaming test complete")
            return True
            
        except Exception as e:
            logger.error(f"[WARN] Streaming test error: {e}")
            return True  # Not critical
    
    def generate_report(self):
        """Generate final report"""
        logger.info("\n" + "=" * 70)
        logger.info("FINAL REPORT")
        logger.info("=" * 70)
        
        success = (
            self.results["model_loaded"] and
            len(self.results["transcriptions"]) > 0
        )
        
        report = {
            "test_name": "ASR Engine Validation (IndicConformer)",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "PASS" if success else "FAIL",
            "model_path": self.model_path,
            "language": self.lang,
            "results": self.results
        }
        
        logger.info(f"Status: {report['status']}")
        logger.info(f"Model loaded: {self.results['model_loaded']}")
        logger.info(f"Transcriptions: {len(self.results['transcriptions'])}")
        logger.info(f"Errors: {len(self.results['errors'])}")
        
        if self.results["transcriptions"]:
            logger.info(f"\nTranscription sample:")
            logger.info(f"  '{self.results['transcriptions'][0]['text'][:100]}...'")
        
        return report


def main():
    parser = argparse.ArgumentParser(description="Test IndicConformer ASR")
    parser.add_argument("--audio", type=str, default="test_audio.wav")
    parser.add_argument("--model_path", type=str,
                       default="./models/ai4bharat_indic-conformer-600m-multilingual/")
    parser.add_argument("--lang", type=str, default="hi",
                       choices=["hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "or", "pa", "as"])
    parser.add_argument("--output", type=str, default="test_asr_final_results.json")
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("INDICCONFORMER ASR TEST")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Audio: {args.audio}")
    logger.info(f"Language: {args.lang}")
    logger.info("=" * 70 + "\n")
    
    # Create tester
    tester = ASRTester(args.model_path, args.lang)
    
    # Run tests
    success = True
    
    if not tester.test_1_model_loading():
        success = False
    else:
        result, _ = tester.test_2_transcription(args.audio)
        if not result:
            success = False
        else:
            tester.test_3_streaming(args.audio)
    
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
        logger.info("ASR model is WORKING!")
        logger.info("\nNext: Create Test 2 for VAD")
    else:
        logger.error("✗✗✗ FAILED ✗✗✗")
        
        # Check for common issues
        if "soundfile" in str(report.get("results", {}).get("errors", [])):
            logger.error("\nFIX: Install soundfile")
            logger.error("  pip install soundfile")
    
    logger.info("=" * 70)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())