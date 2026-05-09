"""
Source Code Editor Tool
------------------------
Permette a JARVIS di modificare il proprio source code in modo sicuro.

Sicurezza:
  - Operazioni consentite: replace (trova-e-sostituisci), append, insert_after.
  - Solo file dentro PROJECT_ROOT, nessun path traversal.
  - Estensioni consentite: .py, .json, .yaml, .yml, .md, .sh.
  - Backup automatico prima di ogni modifica in data/backups/.
  - Nessuna eval/exec del codice modificato: JARVIS non esegue le patch,
    le scrive su disco — sarà l'utente a decidere se riavviare.
  - Ogni modifica viene loggata in logs/source_edits.jsonl.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import ClassVar, Dict, Any, Optional

from tools.base_tool import BaseTool, ToolResult
from core.logger import get_logger

logger = get_logger(__name__)

# Estensioni su cui è consentita la scrittura
ALLOWED_EXTENSIONS = {".py", ".json", ".yaml", ".yml", ".md", ".sh", ".txt"}

# Directory assoluta root del progetto (due livelli su rispetto a tools/)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Log delle modifiche al sorgente
EDIT_LOG = PROJECT_ROOT / "logs" / "source_edits.jsonl"
BACKUP_DIR = PROJECT_ROOT / "data" / "backups"


def _log_edit(record: dict) -> None:
    EDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(EDIT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _backup(path: Path) -> Optional[Path]:
    """Crea un backup del file prima di modificarlo."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    dest = BACKUP_DIR / f"{path.stem}_{ts}{path.suffix}"
    try:
        shutil.copy2(path, dest)
        return dest
    except Exception as e:
        logger.warning(f"Backup fallito per {path}: {e}")
        return None


def _safe_resolve(relative_path: str) -> Optional[Path]:
    """
    Risolve un percorso relativo a PROJECT_ROOT.
    Restituisce None se il percorso è fuori da PROJECT_ROOT o usa '..'
    """
    if ".." in relative_path:
        return None
    target = (PROJECT_ROOT / relative_path).resolve()
    if not str(target).startswith(str(PROJECT_ROOT)):
        return None
    if target.suffix not in ALLOWED_EXTENSIONS:
        return None
    return target


# ---------------------------------------------------------------------------
# Tool: leggi un file sorgente
# ---------------------------------------------------------------------------

class ReadSourceTool(BaseTool):
    """Leggi il contenuto di un file sorgente del progetto."""
    name: ClassVar[str] = "read_source"
    description: ClassVar[str] = (
        "Leggi il contenuto di un file sorgente di JARVIS (percorso relativo alla root del progetto)"
    )
    parameters: ClassVar[Dict[str, Any]] = {
        "type": "object",
        "required": ["path"],
        "properties": {
            "path": {
                "type": "string",
                "description": "Percorso relativo al progetto, es. 'core/orchestrator.py'"
            },
            "start_line": {
                "type": "integer",
                "description": "Prima riga da leggere (1-based, opzionale)"
            },
            "end_line": {
                "type": "integer",
                "description": "Ultima riga da leggere (1-based, opzionale)"
            }
        }
    }

    async def execute(self, action: dict) -> ToolResult:
        path = _safe_resolve(action.get("path", ""))
        if path is None:
            return ToolResult.fail("Percorso non valido o estensione non consentita.")
        if not path.exists():
            return ToolResult.fail(f"File non trovato: {action['path']}")

        try:
            lines = path.read_text(encoding="utf-8").splitlines()
            start = int(action.get("start_line", 1)) - 1
            end   = int(action.get("end_line", len(lines)))
            selected = lines[max(0, start):end]
            numbered = "\n".join(f"{start+i+1:4d} | {l}" for i, l in enumerate(selected))
            return ToolResult.ok(
                f"📄 {action['path']} (righe {start+1}-{start+len(selected)}):\n\n{numbered}",
                data={"lines": selected, "total_lines": len(lines)}
            )
        except Exception as e:
            return ToolResult.fail(f"Errore lettura: {e}")


# ---------------------------------------------------------------------------
# Tool: modifica un file sorgente
# ---------------------------------------------------------------------------

class EditSourceTool(BaseTool):
    """
    Modifica un file sorgente di JARVIS.

    Operazioni supportate:
      - replace   → sostituisce old_text con new_text (prima occorrenza)
      - replace_all → sostituisce tutte le occorrenze
      - append    → aggiunge new_text alla fine del file
      - insert_after → inserisce new_text dopo la riga contenente anchor_text
    """
    name: ClassVar[str] = "edit_source"
    description: ClassVar[str] = (
        "Modifica un file sorgente di JARVIS: "
        "replace, replace_all, append o insert_after."
    )
    requires_confirmation: ClassVar[bool] = True
    parameters: ClassVar[Dict[str, Any]] = {
        "type": "object",
        "required": ["path", "operation"],
        "properties": {
            "path": {
                "type": "string",
                "description": "Percorso relativo al progetto, es. 'core/main.py'"
            },
            "operation": {
                "type": "string",
                "enum": ["replace", "replace_all", "append", "insert_after"],
                "description": "Tipo di operazione"
            },
            "old_text": {
                "type": "string",
                "description": "Testo da cercare (obbligatorio per replace/replace_all/insert_after)"
            },
            "new_text": {
                "type": "string",
                "description": "Testo sostitutivo o da inserire"
            },
            "reason": {
                "type": "string",
                "description": "Motivazione della modifica (per il log)"
            }
        }
    }

    async def execute(self, action: dict) -> ToolResult:
        rel_path  = action.get("path", "")
        operation = action.get("operation", "")
        old_text  = action.get("old_text", "")
        new_text  = action.get("new_text", "")
        reason    = action.get("reason", "self-improvement")

        path = _safe_resolve(rel_path)
        if path is None:
            return ToolResult.fail("Percorso non valido o estensione non consentita.")

        # Il file deve già esistere per replace/insert_after
        if operation in ("replace", "replace_all", "insert_after") and not path.exists():
            return ToolResult.fail(f"File non trovato: {rel_path}")

        # Backup
        backup_path: Optional[Path] = None
        if path.exists():
            backup_path = _backup(path)

        try:
            original = path.read_text(encoding="utf-8") if path.exists() else ""

            # --- OPERAZIONI ---
            if operation == "replace":
                if old_text not in original:
                    return ToolResult.fail(
                        f"Testo da sostituire non trovato in {rel_path}.\n"
                        f"Cerca: {old_text[:120]!r}"
                    )
                result_text = original.replace(old_text, new_text, 1)

            elif operation == "replace_all":
                if old_text not in original:
                    return ToolResult.fail(f"Testo non trovato in {rel_path}.")
                count = original.count(old_text)
                result_text = original.replace(old_text, new_text)
                logger.info(f"replace_all: {count} occorrenze sostituite in {rel_path}")

            elif operation == "append":
                result_text = original + ("\n" if original and not original.endswith("\n") else "") + new_text

            elif operation == "insert_after":
                if old_text not in original:
                    return ToolResult.fail(f"Anchor non trovato in {rel_path}: {old_text[:80]!r}")
                result_text = original.replace(old_text, old_text + "\n" + new_text, 1)

            else:
                return ToolResult.fail(f"Operazione sconosciuta: {operation!r}")

            # Scrivi il file modificato
            path.write_text(result_text, encoding="utf-8")

            # Log della modifica
            _log_edit({
                "timestamp": datetime.utcnow().isoformat(),
                "path": rel_path,
                "operation": operation,
                "reason": reason,
                "backup": str(backup_path) if backup_path else None,
                "chars_before": len(original),
                "chars_after": len(result_text),
            })

            msg = (
                f"✅ {operation} applicato su {rel_path}\n"
                f"   Backup: {backup_path.name if backup_path else 'nessuno'}\n"
                f"   Riavvia JARVIS per caricare le modifiche."
            )
            logger.info(msg)
            return ToolResult.ok(msg, data={"path": rel_path, "operation": operation})

        except Exception as e:
            logger.error(f"edit_source fallito su {rel_path}: {e}", exc_info=True)
            return ToolResult.fail(f"Errore durante la modifica: {e}")


# ---------------------------------------------------------------------------
# Tool: elenca i file sorgente del progetto
# ---------------------------------------------------------------------------

class ListSourceTool(BaseTool):
    """Elenca tutti i file sorgente modificabili del progetto JARVIS."""
    name: ClassVar[str] = "list_source"
    description: ClassVar[str] = (
        "Elenca i file sorgente del progetto JARVIS che possono essere letti o modificati."
    )
    parameters: ClassVar[Dict[str, Any]] = {
        "type": "object",
        "properties": {
            "directory": {
                "type": "string",
                "description": "Sotto-directory da listare (es. 'core', 'tools'). Default: tutto."
            }
        }
    }

    async def execute(self, action: dict) -> ToolResult:
        subdir = action.get("directory", "")
        base   = PROJECT_ROOT / subdir if subdir else PROJECT_ROOT

        if not base.exists() or not base.is_dir():
            return ToolResult.fail(f"Directory non trovata: {subdir}")

        # Esclude cache, backup, __pycache__, .git
        EXCLUDE = {"__pycache__", ".git", ".venv", "venv", "node_modules", "backups"}

        files = []
        for p in sorted(base.rglob("*")):
            if any(part in EXCLUDE for part in p.parts):
                continue
            if p.suffix in ALLOWED_EXTENSIONS and p.is_file():
                rel = p.relative_to(PROJECT_ROOT)
                files.append(str(rel))

        if not files:
            return ToolResult.ok("Nessun file trovato.")

        return ToolResult.ok(
            f"File sorgente ({len(files)}):\n" + "\n".join(f"  {f}" for f in files),
            data=files
        )


# ---------------------------------------------------------------------------
# Tool: rollback dell'ultima modifica a un file
# ---------------------------------------------------------------------------

class RollbackSourceTool(BaseTool):
    """Ripristina un file sorgente dall'ultimo backup disponibile."""
    name: ClassVar[str] = "rollback_source"
    description: ClassVar[str] = (
        "Ripristina un file sorgente di JARVIS dall'ultimo backup automatico."
    )
    requires_confirmation: ClassVar[bool] = True
    parameters: ClassVar[Dict[str, Any]] = {
        "type": "object",
        "required": ["path"],
        "properties": {
            "path": {
                "type": "string",
                "description": "Percorso relativo del file da ripristinare"
            }
        }
    }

    async def execute(self, action: dict) -> ToolResult:
        rel_path = action.get("path", "")
        path     = _safe_resolve(rel_path)
        if path is None:
            return ToolResult.fail("Percorso non valido.")

        stem = path.stem
        suffix = path.suffix
        backups = sorted(BACKUP_DIR.glob(f"{stem}_*{suffix}"), reverse=True)

        if not backups:
            return ToolResult.fail(f"Nessun backup trovato per {rel_path}")

        latest = backups[0]
        try:
            shutil.copy2(latest, path)
            _log_edit({
                "timestamp": datetime.utcnow().isoformat(),
                "path": rel_path,
                "operation": "rollback",
                "backup_used": str(latest),
            })
            return ToolResult.ok(
                f"✅ Rollback completato: {rel_path} ← {latest.name}\n"
                f"   Riavvia JARVIS per caricare la versione precedente."
            )
        except Exception as e:
            return ToolResult.fail(f"Rollback fallito: {e}")
