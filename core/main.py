#!/usr/bin/env python3

import sys
import asyncio
import argparse
import signal
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.orchestrator import Orchestrator
from core.config_loader import ConfigLoader
from core.logger import setup_logging, get_logger

logger = get_logger(__name__)

_TTS_ENGINE = None


# -------------------------
# TTS
# -------------------------

def _speak(text: str, config: ConfigLoader):
    global _TTS_ENGINE

    if not config.get("ui.voice_enabled", False):
        return

    try:
        import pyttsx3

        if _TTS_ENGINE is None:
            _TTS_ENGINE = pyttsx3.init()
            _TTS_ENGINE.setProperty("rate", 185)

        _TTS_ENGINE.say(text)
        _TTS_ENGINE.runAndWait()

    except Exception as e:
        logger.warning(f"TTS error: {e}")


# -------------------------
# MICROFONO (FIX DEFINITIVO)
# -------------------------

def _listen_from_microphone(config: ConfigLoader, symbol: str) -> str:
    """
    Stable Windows audio capture:
    - NO PyAudio
    - NO device index
    - NO WASAPI issues
    """

    try:
        import sounddevice as sd
        import numpy as np
        import speech_recognition as sr
    except Exception as e:
        logger.error(f"Missing deps: {e}")
        return ""

    duration = float(config.get("ui.stt_phrase_limit", 8))
    language = config.get("ui.stt_language", "it-IT")

    recognizer = sr.Recognizer()

    try:
        print(f"{symbol}[voice] ascolto...")

        # IMPORTANT: force stable backend
        sd.default.hostapi = 0  # MME (most stable on Windows)
        sd.default.device = None  # AUTO SELECT

        samplerate = 44100

        recording = sd.rec(
            int(duration * samplerate),
            samplerate=samplerate,
            channels=1,
            dtype="float32",
            blocking=True
        )

        audio = recording.flatten()

        audio = np.clip(audio, -1, 1)
        audio_bytes = (audio * 32767).astype(np.int16).tobytes()

        audio_data = sr.AudioData(audio_bytes, samplerate, 2)

    except Exception as e:
        logger.error(f"Mic capture failed: {e}")
        return ""

    try:
        return recognizer.recognize_google(
            audio_data,
            language=language
        ).strip()

    except Exception as e:
        logger.warning(f"STT failed: {e}")
        return ""


# -------------------------
# LOOP
# -------------------------

def _extract_hotword(text: str, config: ConfigLoader) -> str:
    if not text:
        return ""

    if not config.get("ui.hotword_enabled", False):
        return text

    hotword = config.get("ui.hotword", "jarvis").lower()

    if hotword not in text.lower():
        return ""

    return text.lower().split(hotword, 1)[-1].strip()


async def run(orchestrator: Orchestrator):
    config = orchestrator.config
    name = config.get("assistant.name", "JARVIS")
    symbol = config.get("ui.prompt_symbol", ">> ")

    print(f"\n{name} ready.\n")

    while True:
        try:
            if config.get("ui.voice_enabled", False):
                raw = _listen_from_microphone(config, symbol)
                user_input = _extract_hotword(raw, config)
            else:
                user_input = input(symbol).strip()

        except (EOFError, KeyboardInterrupt):
            print("\nShutdown")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit"]:
            break

        response = await orchestrator.process(user_input)

        print(f"{name}: {response}")
        _speak(response, config)


# -------------------------
# MAIN
# -------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/config.json")
    p.add_argument("--voice", action="store_true")
    return p.parse_args()


async def main():
    args = parse_args()

    config = ConfigLoader(Path(args.config))
    config.set("ui.voice_enabled", args.voice)

    setup_logging(level="INFO")

    orchestrator = Orchestrator(config)
    await orchestrator.initialize()

    try:
        await run(orchestrator)
    finally:
        await orchestrator.shutdown()


if __name__ == "__main__":
    asyncio.run(main())