import asyncio
import logging
import json
import base64
import io
import tempfile
import os
import wave
import time
import array
import math
from typing import Optional, Callable, List, Dict, Any

try:
    from openai import AsyncOpenAI
except ImportError:
    raise ImportError(
        "openai package not found. Please install it with: pip install openai"
    )

from app.config import OPENAI_API_KEY

logger = logging.getLogger(__name__)

class TranscriptionService:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.audio_buffer = []
        self.buffer_size = 0
        self.is_processing = False
        self.processing_task = None
        self.is_active = False
        self.cumulative_transcript = ""
        self.min_buffer_size = 4000
        self.max_buffer_size = 100000
        self.last_processing_time = 0
        self.processing_interval = 0.15
        self.amplitude_threshold = 0.01
        self.silence_threshold = 1.5 
        self.last_speech_time = 0
        self.silence_start_time = None
        self.auto_stop_enabled = True
        
    async def create_connection(self, on_message_callback: Callable, on_error_callback: Callable):
        try:
            self._on_message_callback = on_message_callback
            self._on_error_callback = on_error_callback
            self.audio_buffer = []
            self.buffer_size = 0
            self.is_processing = False
            self.cumulative_transcript = ""
            self.last_processing_time = 0
            self.is_active = True
            self.last_speech_time = time.time()
            self.silence_start_time = None
            self.processing_task = asyncio.create_task(self._processing_loop())
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Whisper service: {e}", exc_info=True)
            return False
    
    async def send_audio(self, audio_data: bytes):
        if not self.is_active:
            return
            
        try:
            self.audio_buffer.append(audio_data)
            self.buffer_size += len(audio_data)
            
            if self.buffer_size >= self.min_buffer_size and not self.is_processing:
                asyncio.create_task(self._process_audio_buffer())
        except Exception as e:
            logger.error(f"Error buffering audio: {e}", exc_info=True)

    async def _processing_loop(self):
        while self.is_active:
            try:
                current_time = time.time()
                
                if self.auto_stop_enabled and self.silence_start_time:
                    silence_duration = current_time - self.silence_start_time
                    if silence_duration >= self.silence_threshold:
                        logger.info(f"Auto-stopping recording after {silence_duration:.1f}s of silence")
                        self.is_active = False
                        try:
                            await self.finalize_stream()
                        except Exception:
                            logger.error("Error finalizing stream during auto-stop", exc_info=True)
                        self.audio_buffer.clear()
                        self.buffer_size = 0
                        await self._send_auto_stop_signal()
                        break
                
                if (self.buffer_size > 0 and 
                    not self.is_processing and 
                    current_time - self.last_processing_time >= self.processing_interval):
                    
                    await self._process_audio_buffer()
                
                await asyncio.sleep(0.05)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in processing loop: {e}", exc_info=True)
                await asyncio.sleep(0.1)

    def _get_audio_amplitude(self, pcm_data: bytes) -> float:
        try:
            int_data = array.array('h', pcm_data)
            
            sum_squares = sum(x * x for x in int_data)
            if not sum_squares or not len(int_data):
                return 0.0
                
            rms = math.sqrt(sum_squares / len(int_data))
            
            normalized_rms = rms / 32768.0  
            
            return normalized_rms
        except Exception as e:
            logger.error(f"Error calculating audio amplitude: {e}", exc_info=True)
            return 0.0

    def _convert_pcm_to_wav(self, pcm_data: bytes, sample_rate: int = 16000, channels: int = 1, bits_per_sample: int = 16) -> bytes:
        try:
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(bits_per_sample // 8)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(pcm_data)
            wav_data = wav_buffer.getvalue()
            wav_buffer.close()
            return wav_data
        except Exception as e:
            logger.error(f"Error converting PCM to WAV: {e}", exc_info=True)
            return pcm_data

    async def _process_audio_buffer(self):
        if not self.audio_buffer or self.is_processing or not self.is_active:
            return
            
        self.is_processing = True
        
        try:
            current_buffer = self.audio_buffer.copy()
            self.audio_buffer = []
            self.buffer_size = 0
            
            combined_audio = b''.join(current_buffer)
            
            if len(combined_audio) < 4000:
                self.is_processing = False
                return
                
            amplitude = self._get_audio_amplitude(combined_audio)
            current_time = time.time()
            
            if amplitude < self.amplitude_threshold:
                if self.silence_start_time is None:
                    self.silence_start_time = current_time
                logger.debug(f"Audio amplitude ({amplitude:.4f}) below threshold ({self.amplitude_threshold:.4f}), silence detected")
                self.is_processing = False
                self.last_processing_time = current_time  
                return
            else:
                self.last_speech_time = current_time
                self.silence_start_time = None  
            
            wav_data = self._convert_pcm_to_wav(combined_audio)
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(wav_data)
                temp_file_path = temp_file.name
            
            try:
                with open(temp_file_path, 'rb') as audio_file:
                    transcript = await self.client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="json",
                        language="en"
                    )
                
                if transcript and transcript.text:
                    await self._send_transcript(transcript.text, is_final=False)
            except Exception as e:
                logger.error(f"Whisper API error: {e}", exc_info=True)
            finally:
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}", exc_info=True)
        finally:
            self.is_processing = False
            self.last_processing_time = time.time()
    
    async def _send_transcript(self, text: str, is_final: bool = False, extra_data: Dict[str, Any] = None):
        try:
            can_send = is_final or self.is_active
            
            if hasattr(self, '_on_message_callback') and self._on_message_callback and can_send:
                payload_text = None

                if text and text.strip():
                    cleaned_text = text.strip()
                    if self.cumulative_transcript:
                        self.cumulative_transcript = f"{self.cumulative_transcript} {cleaned_text}".strip()
                    else:
                        self.cumulative_transcript = cleaned_text
                    payload_text = self.cumulative_transcript
                elif is_final:
                    payload_text = self.cumulative_transcript

                if payload_text is not None or is_final:
                    await self._on_message_callback(is_final, payload_text or "", extra_data)
        except Exception as e:
            logger.error(f"Error sending transcript: {e}", exc_info=True)
    
    async def _send_auto_stop_signal(self):
        """Send auto-stop signal to client"""
        try:
            if hasattr(self, '_on_message_callback') and self._on_message_callback:
                message = {
                    "action": "auto_stop",
                    "text": "",
                    "is_final": True,
                    "reason": "silence_detected"
                }
                await self._on_message_callback(True, "", message)
        except Exception as e:
            logger.error(f"Error sending auto-stop signal: {e}", exc_info=True)
    
    async def finalize_stream(self):
        try:
            if self.audio_buffer and len(self.audio_buffer) > 0:
                await self._process_audio_buffer()
            
            await self._send_transcript("", is_final=True)
        except Exception as e:
            logger.error(f"Error finalizing stream: {e}", exc_info=True)
    
    async def close_connection(self):
        try:
            await self.finalize_stream()
            
            self.is_active = False
            
            if self.processing_task and not self.processing_task.done():
                self.processing_task.cancel()
                try:
                    await self.processing_task
                except asyncio.CancelledError:
                    pass
        except Exception as e:
            logger.error(f"Error closing Whisper service: {e}", exc_info=True)
        finally:
            self.audio_buffer = []
            self.buffer_size = 0
            self.is_processing = False
            self.processing_task = None
            self.is_active = False

def get_transcription_service() -> TranscriptionService:
    return TranscriptionService()
