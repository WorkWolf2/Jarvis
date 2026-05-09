"""
Core Orchestrator — versione aggiornata
---------------------------------------
Miglioramenti rispetto all'originale:
  - Timeout LLM ridotto di default (config llm.read_timeout).
  - Sistema prompt aggiornato con i nuovi tool read_source / edit_source /
    list_source / rollback_source → JARVIS può correggere il proprio codice.
  - _parse_response più robusto: gestisce JSON embedded in markdown e array
    con lunghezza > 0.
  - process() restituisce subito un messaggio se il LLM è irraggiungibile,
    senza aspettare tutti i retry.
"""

import json
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


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """Sei {name}, un assistente AI locale avanzato in stile Iron Man.

Sei preciso, veloce e capace di eseguire azioni sul sistema.
Hai accesso a tool che ti permettono di controllare il computer, cercare sul web
e — cosa più importante — leggere e modificare il tuo stesso codice sorgente.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMATO RISPOSTA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Azione singola   → {{"type": "nome_azione", "param": "valore"}}
• Azioni in catena → [{{"type": "azione1", ...}}, {{"type": "azione2", ...}}]
• Solo conversazione → testo in linguaggio naturale
• NON mescolare mai JSON e testo nella stessa risposta.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AZIONI DISPONIBILI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{tool_descriptions}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AUTO-MIGLIORAMENTO (usa questi tool per correggere il tuo stesso codice)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Workflow consigliato per modificare il codice:
  1. list_source  → individua i file rilevanti
  2. read_source  → leggi il file interessato (anche con start_line/end_line)
  3. edit_source  → applica la modifica (operation: replace/append/insert_after)
  4. rollback_source → in caso di errore, ripristina il backup automatico

IMPORTANTE: dopo ogni edit_source avvisa il signore che deve riavviare JARVIS
per caricare le modifiche.

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
6. Per info aggiornate usa web_search con riferimenti alle fonti.
7. Quando modifichi il codice sorgente, descrivi SEMPRE cosa hai cambiato e perché.
"""


class Orchestrator:
    """Orchestratore centrale: coordina LLM, tool, memoria e self-improvement."""

    def __init__(self, config: ConfigLoader) -> None:
        self.config = config
        self._initialized = False

        self.llm: Optional[OllamaClient] = None
        self.memory: Optional[MemoryManager] = None
        self.tools: Optional[ToolRegistry] = None
        self.router: Optional[ActionRouter] = None
        self.safety: Optional[SafetyValidator] = None
        self.interaction_logger: Optional[InteractionLogger] = None
        self.analyzer: Optional[SelfImprovementAnalyzer] = None
        self.patch_manager: Optional[PatchManager] = None

        self._interaction_count = 0
        self._session_start = datetime.utcnow()

    # ------------------------------------------------------------------
    # Inizializzazione
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        logger.info("Inizializzazione sottosistemi JARVIS...")
        root = Path(__file__).parent.parent

        self.safety = SafetyValidator(self.config)
        logger.info("✓ Safety validator")

        db_path = root / self.config.get("memory.db_path", "data/jarvis_memory.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.memory = MemoryManager(db_path, self.config)
        await self.memory.initialize()
        logger.info("✓ Memory system")

        self.tools = ToolRegistry(self.config)
        self.tools.auto_discover(root / "tools")
        logger.info(f"✓ Tool registry ({len(self.tools.list_tools())} tool caricati)")

        self.llm = OllamaClient(self.config)
        model_ok = await self.llm.check_connection()
        if not model_ok:
            logger.warning(
                f"Modello '{self.config.get('llm.model')}' non disponibile. "
                "Avvia Ollama: ollama serve"
            )
        else:
            logger.info(f"✓ LLM ({self.config.get('llm.model')})")

        self.router = ActionRouter(self.tools, self.safety, self.config)
        logger.info("✓ Action router")

        log_path = root / self.config.get("self_improve.log_path", "logs/interactions.jsonl")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.interaction_logger = InteractionLogger(log_path)
        logger.info("✓ Interaction logger")

        self.analyzer = SelfImprovementAnalyzer(
            self.interaction_logger, self.llm, self.config
        )

        patches_dir = root / "data" / "patches"
        patches_dir.mkdir(parents=True, exist_ok=True)
        self.patch_manager = PatchManager(patches_dir, self.config)
        logger.info("✓ Self-improvement engine")

        self._initialized = True
        logger.info("JARVIS completamente inizializzato ✓")

    # ------------------------------------------------------------------
    # Pipeline principale
    # ------------------------------------------------------------------

    async def process(self, user_input: str) -> str:
        if not self._initialized:
            return "Sistema non inizializzato. Chiama initialize() prima."

        self._interaction_count += 1
        iid = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{self._interaction_count}"

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

            logger.debug(f"LLM raw: {raw_response[:200]}")

            parsed = self._parse_response(raw_response)
            final_response = raw_response
            execution_success = True

            if parsed is not None:
                # Supporta catene multi-step (l'LLM può emettere più azioni)
                result, execution_success = await self.router.route(parsed, user_input)
                final_response = result

            await self.memory.add_message("user", user_input)
            await self.memory.add_message("assistant", final_response)

            await self.interaction_logger.log({
                "id": iid,
                "timestamp": datetime.utcnow().isoformat(),
                "user_input": user_input,
                "llm_response": raw_response,
                "parsed_action": parsed if isinstance(parsed, (dict, list)) else str(parsed),
                "final_response": final_response,
                "success": execution_success,
                "model": self.config.get("llm.model"),
            })

            await self._maybe_run_analysis()
            return final_response

        except Exception as e:
            logger.error(f"Errore processing: {e}", exc_info=True)
            error_msg = f"Errore nell'elaborazione della richiesta: {e}"
            await self.interaction_logger.log({
                "id": iid,
                "timestamp": datetime.utcnow().isoformat(),
                "user_input": user_input,
                "error": str(e),
                "success": False,
            })
            return error_msg

    # ------------------------------------------------------------------
    # Parsing risposta LLM
    # ------------------------------------------------------------------

    def _parse_response(self, response: str) -> Optional[Any]:
        """
        Estrae JSON action(s) dalla risposta LLM.
        Gestisce:
          - JSON puro  { ... }  o  [ ... ]
          - JSON dentro blocchi markdown ```json ... ```
          - JSON embedded in testo misto
        """
        response = response.strip()

        # 1. Prova diretta
        if response.startswith(("{", "[")):
            parsed = self._try_json(response)
            if parsed is not None:
                return parsed

        # 2. Estrai da blocco markdown
        import re
        md_block = re.search(r"```(?:json)?\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```", response)
        if md_block:
            parsed = self._try_json(md_block.group(1))
            if parsed is not None:
                return parsed

        # 3. Cerca il primo oggetto JSON con "type"
        json_obj = re.search(r'\{[^{}]*"type"\s*:[^{}]*\}', response, re.DOTALL)
        if json_obj:
            parsed = self._try_json(json_obj.group(0))
            if parsed is not None:
                return parsed

        return None  # testo puro

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
    # Costruzione system prompt
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

    async def startup_message(self) -> None:
        name  = self.config.get("assistant.name", "JARVIS")
        model = self.config.get("llm.model", "sconosciuto")
        n     = len(self.tools.list_tools()) if self.tools else 0
        print(f"{name}: Tutti i sistemi online. Modello: {model}. {n} tool caricati.")

    async def _maybe_run_analysis(self) -> None:
        if not self.config.get("self_improve.enabled", True):
            return
        min_i = self.config.get("self_improve.min_interactions_before_analysis", 10)
        if self._interaction_count % max(min_i, 1) == 0:
            logger.info("Pianificazione analisi self-improvement in background...")
            asyncio.create_task(self._background_analysis())

    async def _background_analysis(self) -> None:
        try:
            patches = await self.analyzer.analyze()
            if patches:
                self.patch_manager.save_patches(patches)
                logger.info(f"Analisi background: {len(patches)} patch generate")
        except Exception as e:
            logger.error(f"Analisi background fallita: {e}")

    async def shutdown(self) -> None:
        logger.info("Spegnimento JARVIS...")
        if self.memory:
            await self.memory.close()
        logger.info("JARVIS spento.")