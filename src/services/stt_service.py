"""
src/services/stt_service.py
─────────────────────────────────────────────────────────────
Speech-to-Text service with two backends:
  1. IBM Watson Speech-to-Text (cloud, preferred if credentials present)
  2. OpenAI Whisper (local fallback)

Usage:
    stt = SpeechToTextService()
    transcript = stt.transcribe(audio_bytes, content_type="audio/wav")
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from src.config import get_settings

logger = logging.getLogger(__name__)


class SpeechToTextService:
    """
    Transcribes audio to text. Automatically selects the backend
    based on whether Watson STT credentials are configured.
    """

    def __init__(self) -> None:
        self._cfg = get_settings()
        self._backend = self._select_backend()

    def _select_backend(self) -> str:
        if self._cfg.watson_stt_apikey and self._cfg.watson_stt_url:
            logger.info("STT backend: IBM Watson Speech-to-Text")
            return "watson"
        logger.info("STT backend: OpenAI Whisper (local)")
        return "whisper"

    # ── Public API ────────────────────────────────────────────

    def transcribe(
        self,
        audio_data: bytes,
        content_type: str = "audio/wav",
        language: str = "en-US",
    ) -> str:
        """
        Transcribe raw audio bytes to text.

        Args:
            audio_data: Raw audio file bytes (WAV, MP3, FLAC, etc.)
            content_type: MIME type of the audio (default: audio/wav)
            language: BCP-47 language code (default: en-US)

        Returns:
            Transcribed text string.
        """
        if self._backend == "watson":
            return self._transcribe_watson(audio_data, content_type, language)
        return self._transcribe_whisper(audio_data, content_type)

    # ── Watson STT backend ────────────────────────────────────

    def _transcribe_watson(
        self, audio_data: bytes, content_type: str, language: str
    ) -> str:
        try:
            from ibm_watson import SpeechToTextV1
            from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

            authenticator = IAMAuthenticator(self._cfg.watson_stt_apikey)
            stt = SpeechToTextV1(authenticator=authenticator)
            stt.set_service_url(self._cfg.watson_stt_url)

            response = stt.recognize(
                audio=audio_data,
                content_type=content_type,
                model=f"{language}_BroadbandModel",
                timestamps=False,
                word_confidence=False,
            ).get_result()

            results = response.get("results", [])
            transcript = " ".join(
                r["alternatives"][0]["transcript"]
                for r in results
                if r.get("final") and r.get("alternatives")
            )
            return transcript.strip()

        except ImportError:
            logger.warning(
                "ibm-watson package not installed; falling back to Whisper."
            )
            return self._transcribe_whisper(audio_data, "audio/wav")
        except Exception as exc:
            logger.error("Watson STT transcription failed: %s", exc)
            raise

    # ── Whisper backend ───────────────────────────────────────

    def _transcribe_whisper(self, audio_data: bytes, content_type: str) -> str:
        try:
            import whisper  # type: ignore

            # Determine file extension from MIME type
            ext_map = {
                "audio/wav": ".wav",
                "audio/wave": ".wav",
                "audio/mpeg": ".mp3",
                "audio/mp3": ".mp3",
                "audio/flac": ".flac",
                "audio/ogg": ".ogg",
                "audio/webm": ".webm",
            }
            ext = ext_map.get(content_type, ".wav")

            # Write to a temp file (Whisper requires a file path)
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(audio_data)
                tmp_path = tmp.name

            model = whisper.load_model(self._cfg.whisper_model_size)
            result = model.transcribe(tmp_path, language="en")

            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)

            return result.get("text", "").strip()

        except ImportError:
            logger.error(
                "openai-whisper is not installed. "
                "Install it with: pip install openai-whisper"
            )
            raise RuntimeError(
                "No STT backend available. Configure Watson credentials or install Whisper."
            )
        except Exception as exc:
            logger.error("Whisper transcription failed: %s", exc)
            raise
