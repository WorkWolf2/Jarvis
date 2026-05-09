#!/usr/bin/env python3
"""
JARVIS Main Entry Point — Versione Ultra-Veloce (Vosk + pyttsx3 ITA)
------------------------------------------------------
Miglioramenti:
    • STT: Vosk Offline (Zero latenza di rete, zero WinError 50).
    • TTS: Selezione automatica voce italiana (Elsa/Cosimo/Ita).
    • VAD: Ridotto tempo di attesa silenzio a 400ms per maggiore reattività.
    • Pulizia: Rimosse conversioni FLAC e richieste HTTP legacy.
"""

import sys
import asyncio
import argparse
import threading
import json
import io
from pathlib import Path

# Gestione path progetto
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.orchestrator import Orchestrator
from core.config_loader import ConfigLoader
from core.logger import setup_logging, get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# STT Offline via Vosk (Velocità massima, no subprocess)
# ---------------------------------------------------------------------------

_VOSK_MODEL = None
_VOSK_LOCK = threading.Lock()

def _get_vosk_model(model_path: str = "vosk-model-it"):
    """Carica il modello Vosk come Singleton."""
    global _VOSK_MODEL
    from vosk import Model, SetLogLevel
    with _VOSK_LOCK:
        if _VOSK_MODEL is None:
            SetLogLevel(-1) # Silenzia i log interni di Vosk
            if not Path(model_path).exists():
                logger.error(f"ERRORE: Cartella modello '{model_path}' non trovata!")
                return None
            _VOSK_MODEL = Model(model_path)
    return _VOSK_MODEL

def _vosk_stt_sync(audio_int16: "np.ndarray", sample_rate: int) -> str:
    """Esegue la trascrizione sui bytes raw PCM."""
    from vosk import KaldiRecognizer
    model = _get_vosk_model()
    if model is None: return ""
    
    rec = KaldiRecognizer(model, sample_rate)
    rec.AcceptWaveform(audio_int16.tobytes())
    result = json.loads(rec.FinalResult())
    return result.get("text", "").strip()

# ---------------------------------------------------------------------------
# VAD + Registrazione (sounddevice)
# ---------------------------------------------------------------------------

def _record_with_vad(config: ConfigLoader, symbol: str):
    """Registra con VAD ottimizzato e calibrazione del rumore ambientale."""
    try:
        import sounddevice as sd
        import numpy as np
    except ImportError:
        logger.error("Dipendenze audio mancanti (sounddevice/numpy)")
        return None, 0

    max_seconds = float(config.get("ui.stt_phrase_limit", 10))
    samplerate  = 16000
    block_ms    = 30
    block_size  = int(samplerate * block_ms / 1000)

    # Parametri ottimizzati
    MAX_SILENCE_BLOCKS = int(400 / block_ms) 
    MIN_SPEECH_BLOCKS  = int(80  / block_ms) 
    max_blocks         = int(max_seconds * 1000 / block_ms)

    frames, speech_blocks, silence_blocks, recording = [], 0, 0, False

    try:
        with sd.InputStream(samplerate=samplerate, channels=1,
                            dtype="float32", blocksize=block_size) as stream:
            
            # --- FASE 1: CALIBRAZIONE RUMORE DI FONDO ---
            print(f"{symbol}[🎤 calibrazione rumore...]", end="\r", flush=True)
            noise_levels = []
            for _ in range(15):  # Ascolta per ~450ms
                block, _ = stream.read(block_size)
                noise_levels.append(float(np.sqrt(np.mean(block ** 2))))
            
            # Imposta la soglia dinamicamente (il doppio del rumore di fondo, minimo 0.005)
            base_noise = max(noise_levels)
            silence_thresh = max(0.005, base_noise * 2.5)
            
            # --- FASE 2: ASCOLTO REALE ---
            print(f"{symbol}[🎤 in ascolto]           ", end="", flush=True)

            for _ in range(max_blocks):
                block, _ = stream.read(block_size)
                rms = float(np.sqrt(np.mean(block ** 2)))

                if rms > silence_thresh:
                    speech_blocks += 1
                    silence_blocks = 0
                    if speech_blocks >= MIN_SPEECH_BLOCKS:
                        if not recording:
                            print(" 🔴", end="", flush=True)
                            recording = True
                        frames.append(block.copy())
                else:
                    if recording:
                        silence_blocks += 1
                        frames.append(block.copy())
                        if silence_blocks >= MAX_SILENCE_BLOCKS:
                            break # Silenzio rilevato, esce dal loop
                    else:
                        speech_blocks = 0
                        
    except Exception as e:
        logger.error(f"Errore stream audio: {e}")
        return None, 0

    print() # Vai a capo quando finisce
    if not frames: return None, samplerate

    audio = np.concatenate(frames, axis=0).flatten()
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767).astype(np.int16), samplerate

# ---------------------------------------------------------------------------
# Pipeline STT
# ---------------------------------------------------------------------------

async def listen_and_transcribe(config: ConfigLoader, symbol: str) -> str:
    """Pipeline STT locale e veloce."""
    loop = asyncio.get_event_loop()

    # 1. Registrazione (Threaded)
    audio_int16, samplerate = await loop.run_in_executor(
        None, _record_with_vad, config, symbol
    )
    if audio_int16 is None or len(audio_int16) == 0:
        return ""

    # 2. Vosk STT (Threaded)
    text = await loop.run_in_executor(
        None, _vosk_stt_sync, audio_int16, samplerate
    )
    
    if text:
        logger.debug(f"STT → {text!r}")
    return text

# ---------------------------------------------------------------------------
# TTS ITA (pyttsx3)
# ---------------------------------------------------------------------------

_TTS_ENGINE = None
_TTS_LOCK   = threading.Lock()

def _speak_sync(text: str, config: ConfigLoader) -> None:
    global _TTS_ENGINE
    try:
        import pyttsx3
        with _TTS_LOCK:
            if _TTS_ENGINE is None:
                _TTS_ENGINE = pyttsx3.init()
                _TTS_ENGINE.setProperty("rate",   config.get("ui.tts_rate", 175))
                _TTS_ENGINE.setProperty("volume", config.get("ui.tts_volume", 1.0))
                
                # Cerca una voce italiana
                voices = _TTS_ENGINE.getProperty("voices")
                for v in voices:
                    v_name = v.name.lower()
                    if "italian" in v_name or "elsa" in v_name or "cosimo" in v_name or "ita" in v_name:
                        _TTS_ENGINE.setProperty("voice", v.id)
                        break
            
            _TTS_ENGINE.say(text)
            _TTS_ENGINE.runAndWait()
    except Exception as e:
        logger.warning(f"TTS error: {e}")

async def speak(text: str, config: ConfigLoader) -> None:
    if not config.get("ui.voice_enabled", False):
        return
    await asyncio.get_event_loop().run_in_executor(None, _speak_sync, text, config)

# ---------------------------------------------------------------------------
# Logica Ausiliaria (Hotword & Main Loop)
# ---------------------------------------------------------------------------

def _extract_after_hotword(text: str, config: ConfigLoader) -> str:
    if not text or not config.get("ui.hotword_enabled", False):
        return text
    hotword = config.get("ui.hotword", "jarvis").lower()
    if hotword not in text.lower():
        return ""
    words = text.split()
    idx = next((i for i, w in enumerate(words) if hotword in w.lower()), -1)
    return " ".join(words[idx + 1:]).strip() if idx >= 0 else text

async def run(orchestrator: Orchestrator) -> None:
    config = orchestrator.config
    name, symbol = config.get("assistant.name", "JARVIS"), config.get("ui.prompt_symbol", ">> ")
    
    print(f"\n{'='*52}\n  {name} — Sistema Attivo (STT Locale)\n{'='*52}\n")

    while True:
        try:
            if config.get("ui.voice_enabled", False):
                raw = await listen_and_transcribe(config, symbol)
                user_input = _extract_after_hotword(raw, config)
                if raw and not user_input: continue
            else:
                user_input = await asyncio.get_event_loop().run_in_executor(None, lambda: input(symbol).strip())
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input: continue
        if user_input.lower() in {"exit", "quit", "esci", "spegniti"}:
            await speak("Arrivederci, signore.", config)
            break

        print("  ⏳ elaborazione...", end="\r")
        response = await orchestrator.process(user_input)
        print(" " * 50, end="\r")
        print(f"{name}: {response}\n")
        await speak(response, config)

async def main() -> None:
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="config/config.json")
    args.add_argument("--voice", action="store_true")
    args.add_argument("--text", action="store_true")
    parsed = args.parse_args()

    config = ConfigLoader(Path(parsed.config))
    if parsed.voice: config.set("ui.voice_enabled", True)
    elif parsed.text: config.set("ui.voice_enabled", False)

    setup_logging(level=config.get("logging.level", "INFO"), log_file=config.get("logging.file", "logs/jarvis.log"))

    orchestrator = Orchestrator(config)
    await orchestrator.initialize()
    try:
        await run(orchestrator)
    finally:
        await orchestrator.shutdown()

if __name__ == "__main__":
    asyncio.run(main())