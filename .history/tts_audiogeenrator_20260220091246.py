#import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from gtts import gTTS
import io
import numpy as np
from typing import Dict, Optional, BinaryIO
import logging
import soundfile as sf

class AudioGenerator:
    """Generates high-quality speech from processed text"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize TTS models
        self.use_neural = config.get('use_neural_tts', True)
        
        if self.use_neural:
            try:
                self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
                self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
                self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
                
                # Load speaker embeddings
                self.speaker_embeddings = torch.randn((1, 512))  # Random voice
                
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.model.to(self.device)
                self.vocoder.to(self.device)
                
            except Exception as e:
                self.logger.warning(f"Neural TTS failed to load: {e}, falling back to gTTS")
                self.use_neural = False
    
    def generate_audio(self, text: str, voice_params: Optional[Dict] = None) -> BinaryIO:
        """
        Generate audio from text
        
        Args:
            text: Text to convert to speech
            voice_params: Voice customization parameters
            
        Returns:
            Audio file as bytes buffer
        """
        try:
            if self.use_neural and len(text) < 500:  # Limit neural TTS to shorter texts
                return self._generate_neural_audio(text, voice_params)
            else:
                return self._generate_gtts_audio(text, voice_params)
                
        except Exception as e:
            self.logger.error(f"Audio generation failed: {e}")
            return self._generate_fallback_audio(text)
    
    def _generate_neural_audio(self, text: str, voice_params: Optional[Dict]) -> BinaryIO:
        """Generate audio using neural TTS (SpeechT5)"""
        # Process input
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        
        # Generate speech
        with torch.no_grad():
            speech = self.model.generate_speech(
                inputs["input_ids"], 
                self.speaker_embeddings.to(self.device),
                vocoder=self.vocoder
            )
        
        # Convert to audio bytes
        audio_bytes = io.BytesIO()
        sf.write(audio_bytes, speech.cpu().numpy(), 16000, format='wav')
        audio_bytes.seek(0)
        
        return audio_bytes
    
    def _generate_gtts_audio(self, text: str, voice_params: Optional[Dict]) -> BinaryIO:
        """Generate audio using Google TTS"""
        # Configure TTS
        tts_params = {
            'text': text,
            'lang': voice_params.get('language', 'en') if voice_params else 'en',
            'slow': voice_params.get('slow', False) if voice_params else False
        }
        
        # Generate audio
        tts = gTTS(**tts_params)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        
        return audio_bytes
    
    def _generate_fallback_audio(self, text: str) -> BinaryIO:
        """Generate simple beep as fallback"""
        # Generate a simple beep
        sample_rate = 22050
        duration = 1.0
        frequency = 440
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        beep = np.sin(2 * np.pi * frequency * t)
        beep = (beep * 32767).astype(np.int16)
        
        audio_bytes = io.BytesIO()
        sf.write(audio_bytes, beep, sample_rate, format='wav')
        audio_bytes.seek(0)
        
        return audio_bytes
    
    def concatenate_audio_segments(self, segments: list) -> BinaryIO:
        """Concatenate multiple audio segments"""
        all_audio = []
        
        for segment in segments:
            segment.seek(0)
            audio, sr = sf.read(segment)
            all_audio.append(audio)
        
        combined = np.concatenate(all_audio)
        
        audio_bytes = io.BytesIO()
        sf.write(audio_bytes, combined, 16000, format='wav')
        audio_bytes.seek(0)
        
        return audio_bytes