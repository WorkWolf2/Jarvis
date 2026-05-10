"""
Core Orchestrator v2
---------------------
Miglioramenti:
  - Traccia la latenza di ogni request e la include nel log.
  - Chiama self_improve.analyzer.record_interaction() dopo ogni risposta
    per metriche live senza attendere il ciclo periodico.
  - Messaggio di errore LLM immediato (no attesa retry inutile visibile).
  - _parse_response più tollerante: accetta JSON con prefisso/suffisso testo.
  - warm-up LLM all'avvio (riduce latenza della prima query).
"""

import json
import time
import asyncio
from datetime import datetime
from typing import Optional, Any
from pathlib import Path

from core.config_loader import ConfigLoader
from core.logger import get_logger
from core.router import ActionRouter
from llm.ollama_client import OllamaClient
from memory.memory import MemoryManager
from tools.registry import ToolRegistry
from self_improve.logger import InteractionLogger
from self_improve.analyzer import SelfImprovementAnalyzer
from self_improve.apply_patch import PatchManager
from safety.validator import SafetyValidator

logger = get_logger(__name__)


SYSTEM_PROMPT_TEMPLATE = """Sei {name}, un assistente AI locale avanzato in stile Iron Man.

Sei preciso, veloce e capace di eseguire azioni sul sistema.
Hai accesso a tool che ti permettono di controllare il computer, cercare sul web,
fare ricerche avanzate con download di file, e modificare il tuo stesso codice sorgente.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMATO RISPOSTA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Azione singola   → {{"type": "nome_azione", "param": "valore"}}
• Azioni in catena → [{{"type": "azione1", ...}}, {{"type": "azione2", ...}}]
• Solo conversazione → testo in linguaggio naturale
• NON mescolare JSON e testo nella stessa risposta.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AZIONI DISPONIBILI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{tool_descriptions}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RICERCA AVANZATA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Per richieste tipo "cerca X e scarica i risultati" o "trova articoli su Y":
  → usa {{"type": "research", "query": "...", "max_browser_tabs": 4}}
Questo tool cerca sul web, scarica gli articoli, apre il browser e genera un report.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AUTO-MIGLIORAMENTO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Workflow per modificare il codice:
  1. list_source  → individua i file
  2. read_source  → leggi il file
  3. edit_source  → applica la modifica
  4. rollback_source → ripristina se errore

Dopo edit_source avvisa il signore che deve riavviare JARVIS.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTESTO SESSIONE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REGOLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Usa solo le azioni nell'elenco sopra.
2. Se non puoi completare una richiesta in modo sicuro, spiegalo in testo.
3. Sii conciso. Risposte brevi e precise.
4. Rispondi sempre nella stessa lingua dell'utente.
5. Nelle risposte conversazionali, rivolgiti all'utente come "signore".
6. Per info aggiornate usa research o web_search con riferimento alle fonti.
7. Quando modifichi il codice sorgente, descrivi SEMPRE cosa hai cambiato e perché.
"""


class Orchestrator:
    """Orchestratore centrale v2 — con latency tracking e self-improve live."""

    def __init__(self, config: ConfigLoader) -> None:
        self.config       = config
        self._initialized = False

        self.llm:                Optional[OllamaClient]          = None
        self.memory:             Optional[MemoryManager]         = None
        self.tools:              Optional[ToolRegistry]          = None
        self.router:             Optional[ActionRouter]          = None
        self.safety:             Optional[SafetyValidator]       = None
        self.interaction_logger: Optional[InteractionLogger]     = None
        self.analyzer:           Optional[SelfImprovementAnalyzer] = None
        self.patch_manager:      Optional[PatchManager]          = None

        self._interaction_count = 0
        self._session_start     = datetime.utcnow()

    # ------------------------------------------------------------------
    # Inizializzazione
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        logger.info("Inizializzazione JARVIS...")
        root = Path(__file__).parent.parent

        self.safety = SafetyValidator(self.config)

        db_path = root / self.config.get("memory.db_path", "data/jarvis_memory.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.memory = MemoryManager(db_path, self.config)
        await self.memory.initialize()

        self.tools = ToolRegistry(self.config)
        self.tools.auto_discover(root / "tools")
        logger.info(f"✓ {len(self.tools.list_tools())} tool caricati: {self.tools.list_tools()}")

        self.llm = OllamaClient(self.config)
        model_ok = await self.llm.check_connection()
        if not model_ok:
            logger.warning("LLM non disponibile. Avvia Ollama: ollama serve")
        else:
            logger.info(f"✓ LLM: {self.config.get('llm.model')}")
            # Warm-up in background
            asyncio.create_task(self._warmup())

        self.router = ActionRouter(self.tools, self.safety, self.config)

        log_path = root / self.config.get("self_improve.log_path", "logs/interactions.jsonl")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.interaction_logger = InteractionLogger(log_path)

        self.analyzer = SelfImprovementAnalyzer(
            self.interaction_logger, self.llm, self.config
        )

        patches_dir = root / "data" / "patches"
        patches_dir.mkdir(parents=True, exist_ok=True)
        self.patch_manager = PatchManager(patches_dir, self.config)

        self._initialized = True
        logger.info("JARVIS pronto ✓")

    async def _warmup(self) -> None:
        """Warm-up silenzioso del modello LLM."""
        try:
            await asyncio.sleep(1)
            await self.llm.generate("ok", max_tokens=1)
            logger.debug("LLM warm-up completato")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Pipeline principale
    # ------------------------------------------------------------------

    async def process(self, user_input: str) -> str:
        if not self._initialized:
            return "Sistema non inizializzato."

        self._interaction_count += 1
        iid = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{self._interaction_count}"
        t_start = time.monotonic()

        logger.debug(f"Input [{iid}]: {user_input[:100]}")

        try:
            context = await self.memory.get_context(
                max_messages=self.config.get("memory.max_context_messages", 20)
            )
            system_prompt = self._build_system_prompt(context)
            history = await self.memory.get_recent_messages(
                n=self.config.get("llm.history_messages", 6)
            )

            raw_response = await self.llm.chat(
                system_prompt=system_prompt,
                user_message=user_input,
                history=history,
            )

            logger.debug(f"LLM raw [{iid}]: {raw_response[:200]}")

            parsed          = self._parse_response(raw_response)
            final_response  = raw_response
            execution_success = True

            if parsed is not None:
                result, execution_success = await self.router.route(parsed, user_input)
                final_response = result

            await self.memory.add_message("user", user_input)
            await self.memory.add_message("assistant", final_response)

            latency = round(time.monotonic() - t_start, 2)

            record = {
                "id": iid,
                "timestamp": datetime.utcnow().isoformat(),
                "user_input": user_input,
                "llm_response": raw_response,
                "parsed_action": parsed if isinstance(parsed, (dict, list)) else str(parsed),
                "final_response": final_response,
                "success": execution_success,
                "latency_seconds": latency,
                "model": self.config.get("llm.model"),
            }

            await self.interaction_logger.log(record)

            # Aggiorna metriche live per self-improvement
            if self.analyzer:
                self.analyzer.record_interaction(record)

            await self._maybe_run_analysis()
            return final_response

        except Exception as e:
            logger.error(f"Errore processing: {e}", exc_info=True)
            latency = round(time.monotonic() - t_start, 2)
            err_msg = f"Errore nell'elaborazione: {e}"
            await self.interaction_logger.log({
                "id": iid,
                "timestamp": datetime.utcnow().isoformat(),
                "user_input": user_input,
                "error": str(e),
                "success": False,
                "latency_seconds": latency,
            })
            return err_msg

    # ------------------------------------------------------------------
    # Parsing risposta LLM — più tollerante
    # ------------------------------------------------------------------

    def _parse_response(self, response: str) -> Optional[Any]:
        """
        Estrae JSON action(s) dalla risposta LLM.
        Più tollerante: trova JSON anche in mezzo al testo.
        """
        response = response.strip()

        # 1. JSON puro
        if response.startswith(("{", "[")):
            parsed = self._try_json(response)
            if parsed is not None:
                return parsed

        # 2. Blocco markdown ```json ... ``` o ``` ... ```
        import re
        md = re.search(r"```(?:json)?\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```", response)
        if md:
            parsed = self._try_json(md.group(1))
            if parsed is not None:
                return parsed

        # 3. JSON con "type" nel testo (cerca il primo oggetto completo)
        # Usa un approccio bracket-matching per JSON annidati
        for start_char, end_char in [('{', '}'), ('[', ']')]:
            idx = response.find(start_char)
            if idx == -1:
                continue
            depth = 0
            for i in range(idx, len(response)):
                if response[i] == start_char:
                    depth += 1
                elif response[i] == end_char:
                    depth -= 1
                    if depth == 0:
                        candidate = response[idx:i+1]
                        parsed = self._try_json(candidate)
                        if parsed is not None:
                            return parsed
                        break

        return None

    @staticmethod
    def _try_json(text: str) -> Optional[Any]:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and "type" in parsed:
                return parsed
            if (
                isinstance(parsed, list)
                and len(parsed) > 0
                and all(isinstance(a, dict) and "type" in a for a in parsed)
            ):
                return parsed
        except json.JSONDecodeError:
            pass
        return None

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def _build_system_prompt(self, context: str) -> str:
        return SYSTEM_PROMPT_TEMPLATE.format(
            name=self.config.get("assistant.name", "JARVIS"),
            tool_descriptions=self.tools.get_descriptions_for_prompt(),
            context=context or "Nessun contesto precedente.",
        )

    # ------------------------------------------------------------------
    # Utilità
    # ------------------------------------------------------------------

    async def _maybe_run_analysis(self) -> None:
        if not self.config.get("self_improve.enabled", True):
            return
        min_i = self.config.get("self_improve.min_interactions_before_analysis", 10)
        if self._interaction_count % max(min_i, 1) == 0:
            asyncio.create_task(self._background_analysis())

    async def _background_analysis(self) -> None:
        try:
            patches = await self.analyzer.analyze()
            if patches:
                self.patch_manager.save_patches(
                    [p for p in patches if not p.get("auto_apply")]
                )
                applied = [p for p in patches if p.get("auto_apply")]
                if applied:
                    logger.info(
                        f"Self-improve: {len(applied)} patch config applicate automaticamente"
                    )
        except Exception as e:
            logger.error(f"Background analysis fallita: {e}")

    async def shutdown(self) -> None:
        logger.info("Spegnimento JARVIS...")
        if self.memory:
            await self.memory.close()
        logger.info("JARVIS spento.")