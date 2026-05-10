#!/usr/bin/env python3
"""
JARVIS Main Entry Point — Versione Ottimizzata
------------------------------------------------------
Miglioramenti principali:
  • HOTWORD: riconosciuta UNA VOLTA → sessione attiva per N secondi di silenzio.
    Dopo la hotword non devi ripeterla: parla liberamente finché JARVIS capisce
    che stai ancora parlando (timeout di inattività configurabile).
  • VELOCITÀ: pipeline STT completamente parallela; il modello Vosk viene
    precargicato in background al boot; il TTS parla mentre il prossimo ciclo
    di ascolto si prepara.
  • VAD migliorato: soglia dinamica più aggressiva, silenzio rilevato più veloce.
  • Gestione robusto degli errori audio senza bloccare il loop principale.
"""

import sys
import asyncio
import argparse
import threading
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.orchestrator import Orchestrator
from core.config_loader import ConfigLoader
from core.logger import setup_logging, get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Thread pool condiviso (evita overhead di creazione thread per ogni request)
# ---------------------------------------------------------------------------
_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="jarvis-audio")

# ---------------------------------------------------------------------------
# Stato sessione hotword
# ---------------------------------------------------------------------------
class SessionState:
    """Traccia se siamo in sessione attiva (hotword già pronunciata)."""

    def __init__(self, timeout_seconds: float = 30.0):
        self.active = False
        self.last_activity = 0.0
        self.timeout = timeout_seconds
        self._lock = threading.Lock()

    def activate(self):
        with self._lock:
            self.active = True
            self.last_activity = time.time()
            logger.debug("Sessione JARVIS attivata")

    def touch(self):
        """Aggiorna timestamp ultima attività."""
        with self._lock:
            if self.active:
                self.last_activity = time.time()

    def deactivate(self):
        with self._lock:
            self.active = False
            logger.debug("Sessione JARVIS disattivata (timeout)")

    @property
    def is_active(self) -> bool:
        with self._lock:
            if not self.active:
                return False
            if time.time() - self.last_activity > self.timeout:
                self.active = False
                return False
            return True

    def check_and_expire(self) -> bool:
        """Ritorna True se la sessione era attiva ma è scaduta."""
        with self._lock:
            if self.active and time.time() - self.last_activity > self.timeout:
                self.active = False
                return True
        return False


# ---------------------------------------------------------------------------
# Vosk STT — Singleton con precaricamento asincrono
# ---------------------------------------------------------------------------

_VOSK_MODEL = None
_VOSK_READY = threading.Event()
_VOSK_LOCK  = threading.Lock()


def _preload_vosk(model_path: str = "vosk-model-it") -> None:
    """Precarica Vosk in background appena il programma parte."""
    global _VOSK_MODEL
    try:
        from vosk import Model, SetLogLevel
        SetLogLevel(-1)
        if not Path(model_path).exists():
            logger.error(f"Modello Vosk non trovato: '{model_path}'")
            _VOSK_READY.set()
            return
        with _VOSK_LOCK:
            if _VOSK_MODEL is None:
                logger.info(f"Caricamento modello Vosk da '{model_path}'...")
                _VOSK_MODEL = Model(model_path)
                logger.info("Modello Vosk pronto ✓")
        _VOSK_READY.set()
    except Exception as e:
        logger.error(f"Errore caricamento Vosk: {e}")
        _VOSK_READY.set()


def _vosk_stt_sync(audio_int16, sample_rate: int) -> str:
    """Trascrizione STT — attende il modello se non ancora pronto."""
    _VOSK_READY.wait(timeout=30)
    if _VOSK_MODEL is None:
        return ""
    try:
        from vosk import KaldiRecognizer
        rec = KaldiRecognizer(_VOSK_MODEL, sample_rate)
        rec.AcceptWaveform(audio_int16.tobytes())
        result = json.loads(rec.FinalResult())
        return result.get("text", "").strip()
    except Exception as e:
        logger.error(f"Errore STT: {e}")
        return ""


# ---------------------------------------------------------------------------
# VAD + Registrazione ottimizzata
# ---------------------------------------------------------------------------

def _record_with_vad(config: ConfigLoader, symbol: str, short_mode: bool = False):
    """
    Registra audio con VAD.

    short_mode=True → phrase_limit ridotto a 6s (usato quando la sessione
    è già attiva e vogliamo risposta rapida).
    """
    try:
        import sounddevice as sd
        import numpy as np
    except ImportError:
        logger.error("sounddevice/numpy mancanti")
        return None, 0

    phrase_limit = 6.0 if short_mode else float(config.get("ui.stt_phrase_limit", 10))
    samplerate   = 16000
    block_ms     = 25                              # blocchi più piccoli = latenza minore
    block_size   = int(samplerate * block_ms / 1000)

    # Parametri VAD aggressivi
    MAX_SILENCE_BLOCKS = int(350 / block_ms)       # 350ms di silenzio = fine frase
    MIN_SPEECH_BLOCKS  = int(60  / block_ms)       # 60ms minimo per considerare speech
    max_blocks         = int(phrase_limit * 1000 / block_ms)

    frames, speech_blocks, silence_blocks, recording = [], 0, 0, False

    try:
        device_idx = config.get("ui.audio_input_device", None)
        stream_kwargs = dict(
            samplerate=samplerate, channels=1,
            dtype="float32", blocksize=block_size
        )
        if device_idx is not None:
            stream_kwargs["device"] = device_idx

        with sd.InputStream(**stream_kwargs) as stream:

            # Calibrazione rumore (10 blocchi = ~250ms, più veloce)
            print(f"\r{symbol}[🔊 calibro...]          ", end="", flush=True)
            noise_samples = []
            for _ in range(10):
                block, _ = stream.read(block_size)
                noise_samples.append(float(__import__("numpy").sqrt(
                    __import__("numpy").mean(block ** 2)
                )))
            base_noise    = max(noise_samples)
            silence_thresh = max(0.004, base_noise * 2.2)

            print(f"\r{symbol}[🎤 ascolto]            ", end="", flush=True)

            for _ in range(max_blocks):
                block, _ = stream.read(block_size)
                import numpy as np
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
                            break
                    else:
                        speech_blocks = 0

    except Exception as e:
        logger.error(f"Errore stream audio: {e}")
        return None, 0

    print()
    if not frames:
        return None, samplerate

    import numpy as np
    audio = np.concatenate(frames, axis=0).flatten()
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767).astype(np.int16), samplerate


# ---------------------------------------------------------------------------
# Pipeline STT asincrona
# ---------------------------------------------------------------------------

async def listen_and_transcribe(config: ConfigLoader, symbol: str, short_mode: bool = False) -> str:
    loop = asyncio.get_event_loop()

    audio_int16, samplerate = await loop.run_in_executor(
        _EXECUTOR, _record_with_vad, config, symbol, short_mode
    )
    if audio_int16 is None or len(audio_int16) == 0:
        return ""

    text = await loop.run_in_executor(_EXECUTOR, _vosk_stt_sync, audio_int16, samplerate)

    if text:
        logger.debug(f"STT → {text!r}")
    return text


# ---------------------------------------------------------------------------
# TTS ottimizzato — pre-inizializzato, non blocca il loop
# ---------------------------------------------------------------------------

_TTS_ENGINE  = None
_TTS_LOCK    = threading.Lock()
_TTS_READY   = threading.Event()


def _init_tts(config: ConfigLoader) -> None:
    """Pre-inizializza pyttsx3 in background al boot."""
    global _TTS_ENGINE
    try:
        import pyttsx3
        with _TTS_LOCK:
            if _TTS_ENGINE is None:
                engine = pyttsx3.init()
                engine.setProperty("rate",   config.get("ui.tts_rate", 185))
                engine.setProperty("volume", config.get("ui.tts_volume", 1.0))
                voices = engine.getProperty("voices")
                for v in voices:
                    v_name = v.name.lower()
                    if any(k in v_name for k in ("italian", "elsa", "cosimo", "ita", "zira")):
                        engine.setProperty("voice", v.id)
                        logger.debug(f"Voce TTS selezionata: {v.name}")
                        break
                _TTS_ENGINE = engine
        _TTS_READY.set()
        logger.info("TTS pronto ✓")
    except Exception as e:
        logger.warning(f"TTS init fallita: {e}")
        _TTS_READY.set()


def _speak_sync(text: str) -> None:
    _TTS_READY.wait(timeout=10)
    if _TTS_ENGINE is None:
        return
    try:
        with _TTS_LOCK:
            _TTS_ENGINE.say(text)
            _TTS_ENGINE.runAndWait()
    except Exception as e:
        logger.warning(f"TTS speak error: {e}")


async def speak(text: str, config: ConfigLoader) -> None:
    if not config.get("ui.voice_enabled", False):
        return
    # Non blocca il loop principale
    await asyncio.get_event_loop().run_in_executor(_EXECUTOR, _speak_sync, text)


# ---------------------------------------------------------------------------
# Logica hotword
# ---------------------------------------------------------------------------

def _contains_hotword(text: str, hotword: str) -> bool:
    """Controlla se il testo contiene la hotword (tollerante a piccole variazioni STT)."""
    if not text:
        return False
    text_lower  = text.lower()
    hotword_lower = hotword.lower()

    # Match diretto
    if hotword_lower in text_lower:
        return True

    # Tolleranza: le prime 3 lettere (STT può storpiare "simo" → "cimo", "zimo"...)
    prefix = hotword_lower[:3]
    words  = text_lower.split()
    return any(w.startswith(prefix) for w in words)


def _extract_after_hotword(text: str, hotword: str) -> str:
    """Estrae il comando dopo la hotword."""
    if not text:
        return ""
    words    = text.split()
    hw_lower = hotword.lower()
    idx = next(
        (i for i, w in enumerate(words) if hw_lower in w.lower() or w.lower().startswith(hw_lower[:3])),
        -1
    )
    if idx >= 0:
        return " ".join(words[idx + 1:]).strip()
    return text.strip()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def run(orchestrator: Orchestrator) -> None:
    config  = orchestrator.config
    name    = config.get("assistant.name", "JARVIS")
    symbol  = config.get("ui.prompt_symbol", ">> ")
    hotword = config.get("ui.hotword", "jarvis").lower()
    hotword_enabled = config.get("ui.hotword_enabled", True)

    # Timeout sessione: quanto tempo resta "attivo" dopo l'ultima frase
    session_timeout = float(config.get("ui.session_timeout_seconds", 25.0))
    session = SessionState(timeout_seconds=session_timeout)

    print(f"\n{'='*56}")
    print(f"  {name} — Sistema Attivo")
    print(f"  Hotword: '{hotword}' | Sessione: {session_timeout}s")
    print(f"{'='*56}\n")

    # Pre-riscalda il modello LLM con una query vuota in background
    asyncio.create_task(_warmup_llm(orchestrator))

    while True:
        try:
            if config.get("ui.voice_enabled", False):
                # ── MODALITÀ VOCALE ───────────────────────────────────────
                in_session = session.is_active

                if in_session:
                    # Sessione attiva: ascolto veloce senza aspettare hotword
                    raw = await listen_and_transcribe(config, symbol, short_mode=True)
                    if not raw:
                        # Niente audio → controlla se la sessione è scaduta
                        if session.check_and_expire():
                            print(f"  [{name} in attesa — pronuncia '{hotword}' per ricominciare]\n")
                        continue
                    user_input = raw
                    session.touch()

                else:
                    # Aspettiamo la hotword
                    raw = await listen_and_transcribe(config, symbol, short_mode=False)
                    if not raw:
                        continue

                    if hotword_enabled and not _contains_hotword(raw, hotword):
                        # Hotword non pronunciata, ignora
                        logger.debug(f"Hotword assente, ignoro: {raw!r}")
                        continue

                    # Hotword rilevata!
                    session.activate()
                    user_input = _extract_after_hotword(raw, hotword) if hotword_enabled else raw

                    if not user_input:
                        # Solo hotword, nessun comando → chiedi cosa vuole
                        await speak(f"Dimmi pure, signore.", config)
                        # Ascolta subito il comando
                        raw2 = await listen_and_transcribe(config, symbol, short_mode=True)
                        if not raw2:
                            continue
                        user_input = raw2
                        session.touch()

            else:
                # ── MODALITÀ TESTO ───────────────────────────────────────
                loop = asyncio.get_event_loop()
                user_input = await loop.run_in_executor(
                    None, lambda: input(f"{symbol}").strip()
                )

        except (EOFError, KeyboardInterrupt):
            print("\nInterruzione ricevuta.")
            break

        if not user_input:
            continue

        # Comandi di uscita
        if user_input.lower() in {"exit", "quit", "esci", "spegniti", "stop"}:
            await speak("Arrivederci, signore.", config)
            break

        # Mostra input riconosciuto
        if config.get("ui.voice_enabled", False):
            print(f"  🎙️  Tu: {user_input}")

        # ── ELABORAZIONE ─────────────────────────────────────────────────
        print(f"  ⏳ {name} sta elaborando...", end="\r", flush=True)
        t0 = time.time()

        response = await orchestrator.process(user_input)

        elapsed = time.time() - t0
        print(" " * 60, end="\r")  # Pulisce la riga
        print(f"  {name}: {response}")
        print(f"  ⚡ {elapsed:.1f}s\n")

        # TTS in background (non blocca il prossimo ciclo di ascolto)
        asyncio.create_task(speak(response, config))

        # Aggiorna sessione
        if config.get("ui.voice_enabled", False):
            session.touch()


async def _warmup_llm(orchestrator: Orchestrator) -> None:
    """Invia una query di warm-up silenziosa all'LLM per pre-caricare il modello."""
    try:
        await asyncio.sleep(2)  # Aspetta che il sistema sia pronto
        if orchestrator.llm and orchestrator.llm._active_model:
            await orchestrator.llm.generate(".", max_tokens=1)
            logger.debug("LLM warm-up completato")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(description="JARVIS AI Assistant")
    parser.add_argument("--config", default="config/config.json")
    parser.add_argument("--voice", action="store_true", help="Forza modalità vocale")
    parser.add_argument("--text",  action="store_true", help="Forza modalità testo")
    parser.add_argument("--vosk-model", default="vosk-model-it", help="Path modello Vosk")
    parsed = parser.parse_args()

    config = ConfigLoader(Path(parsed.config))
    if parsed.voice:
        config.set("ui.voice_enabled", True)
    elif parsed.text:
        config.set("ui.voice_enabled", False)

    setup_logging(
        level=config.get("logging.level", "INFO"),
        log_file=config.get("logging.file", "logs/jarvis.log")
    )

    # Pre-carica Vosk e TTS in background (parallelizza il boot)
    if config.get("ui.voice_enabled", False):
        vosk_thread = threading.Thread(
            target=_preload_vosk, args=(parsed.vosk_model,), daemon=True
        )
        tts_thread = threading.Thread(
            target=_init_tts, args=(config,), daemon=True
        )
        vosk_thread.start()
        tts_thread.start()

    orchestrator = Orchestrator(config)
    await orchestrator.initialize()

    try:
        await run(orchestrator)
    finally:
        await orchestrator.shutdown()
        _EXECUTOR.shutdown(wait=False)


if __name__ == "__main__":
    asyncio.run(main())