"""
Audio Converter for ASR Testing
=================================

Converts your video/audio file to the correct format for IndicConformer:
- WAV format
- 16000 Hz sample rate
- Mono (single channel)

Usage:
    python convert_audio.py --input 1.mp4 --output test_audio.wav
    
Or just:
    python convert_audio.py  # Uses defaults
"""

import argparse
import sys
from pathlib import Path

def convert_audio(input_path: str, output_path: str, target_sr: int = 16000):
    """Convert audio to WAV format with target sample rate"""
    
    print("=" * 60)
    print("Audio Converter for ASR Testing")
    print("=" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Target: {target_sr} Hz, Mono, WAV")
    print("=" * 60)
    
    # Check if input exists
    if not Path(input_path).exists():
        print(f"\n❌ ERROR: Input file not found: {input_path}")
        print("\nPlease provide a valid input file:")
        print("  python convert_audio.py --input your_file.mp4")
        return False
    
    try:
        # Try Method 1: pydub (easier, works with more formats)
        print("\nMethod 1: Trying pydub...")
        try:
            from pydub import AudioSegment
            
            print("Loading audio file...")
            audio = AudioSegment.from_file(input_path)
            
            print(f"Original: {audio.frame_rate}Hz, {audio.channels} channel(s)")
            
            # Convert to mono
            if audio.channels > 1:
                print("Converting to mono...")
                audio = audio.set_channels(1)
            
            # Resample
            if audio.frame_rate != target_sr:
                print(f"Resampling to {target_sr}Hz...")
                audio = audio.set_frame_rate(target_sr)
            
            # Export as WAV
            print(f"Exporting to {output_path}...")
            audio.export(output_path, format="wav")
            
            print("\n✓ SUCCESS! Audio converted using pydub")
            print_audio_info(output_path)
            return True
            
        except ImportError:
            print("  pydub not installed, trying alternative method...")
            pass
        
        # Method 2: torchaudio (good for PyTorch users)
        print("\nMethod 2: Trying torchaudio...")
        try:
            import torchaudio
            import torch
            
            print("Loading audio file...")
            waveform, sample_rate = torchaudio.load(input_path)
            
            print(f"Original: {sample_rate}Hz, {waveform.shape[0]} channel(s)")
            
            # Convert to mono
            if waveform.shape[0] > 1:
                print("Converting to mono...")
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Resample
            if sample_rate != target_sr:
                print(f"Resampling to {target_sr}Hz...")
                resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
                waveform = resampler(waveform)
            
            # Save as WAV
            print(f"Saving to {output_path}...")
            torchaudio.save(output_path, waveform, target_sr)
            
            print("\n✓ SUCCESS! Audio converted using torchaudio")
            print_audio_info(output_path)
            return True
            
        except ImportError:
            print("  torchaudio not installed, trying alternative method...")
            pass
        
        # Method 3: soundfile + librosa
        print("\nMethod 3: Trying soundfile + librosa...")
        try:
            import soundfile as sf
            import librosa
            
            print("Loading audio file...")
            audio, sr = librosa.load(input_path, sr=target_sr, mono=True)
            
            print(f"Loaded: {sr}Hz, mono, {len(audio)} samples")
            
            # Save as WAV
            print(f"Saving to {output_path}...")
            sf.write(output_path, audio, sr)
            
            print("\n✓ SUCCESS! Audio converted using soundfile")
            print_audio_info(output_path)
            return True
            
        except ImportError:
            print("  soundfile/librosa not installed")
            pass
        
        # Method 4: ffmpeg (requires external tool)
        print("\nMethod 4: Trying ffmpeg (external tool)...")
        print("This requires ffmpeg to be installed on your system")
        print("Download from: https://ffmpeg.org/download.html")
        
        import subprocess
        
        cmd = [
            "ffmpeg",
            "-i", input_path,
            "-ar", str(target_sr),  # Sample rate
            "-ac", "1",             # Mono
            "-y",                   # Overwrite
            output_path
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("\n✓ SUCCESS! Audio converted using ffmpeg")
            print_audio_info(output_path)
            return True
        else:
            print(f"  ffmpeg failed: {result.stderr}")
        
        # All methods failed
        print("\n❌ FAILED: Could not convert audio")
        print("\nPlease install one of these:")
        print("  1. pip install pydub (recommended)")
        print("  2. pip install torchaudio (if using PyTorch)")
        print("  3. pip install soundfile librosa")
        print("  4. Install ffmpeg from https://ffmpeg.org/")
        
        return False
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False


def print_audio_info(audio_path: str):
    """Print info about the converted audio"""
    
    try:
        import wave
        
        with wave.open(audio_path, 'rb') as wav_file:
            print("\n📊 Output File Info:")
            print(f"  Format: WAV")
            print(f"  Sample Rate: {wav_file.getframerate()} Hz")
            print(f"  Channels: {wav_file.getnchannels()}")
            print(f"  Sample Width: {wav_file.getsampwidth()} bytes")
            print(f"  Duration: {wav_file.getnframes() / wav_file.getframerate():.2f} seconds")
            print(f"  Size: {Path(audio_path).stat().st_size / 1024:.2f} KB")
            
    except Exception as e:
        print(f"\n  (Could not read file info: {e})")
    
    print(f"\n✓ File saved: {audio_path}")
    print("  You can now use this file for ASR testing!")


def main():
    parser = argparse.ArgumentParser(description="Convert audio for ASR testing")
    parser.add_argument("--input", "-i", type=str, default="1.mp4",
                       help="Input audio/video file")
    parser.add_argument("--output", "-o", type=str, default="test_audio.wav",
                       help="Output WAV file")
    parser.add_argument("--sample_rate", "-sr", type=int, default=16000,
                       help="Target sample rate (default: 16000)")
    
    args = parser.parse_args()
    
    success = convert_audio(args.input, args.output, args.sample_rate)
    
    if success:
        print("\n" + "=" * 60)
        print("NEXT STEP: Run the ASR test")
        print("=" * 60)
        print(f"  python test_01_asr_real.py --audio {args.output}")
        print("=" * 60)
        return 0
    else:
        print("\n" + "=" * 60)
        print("FAILED: Could not convert audio")
        print("=" * 60)
        print("Please install dependencies or use ffmpeg")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())