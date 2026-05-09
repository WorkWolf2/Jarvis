"""
Safety Validator — versione aggiornata
---------------------------------------
Aggiunta rispetto all'originale:
  - _validate_edit_source: permette le operazioni di auto-modifica del codice
    SOLO sui file dentro PROJECT_ROOT e con le estensioni consentite.
  - _validate_read_source / _validate_list_source / _validate_rollback_source:
    sempre consentiti (read-only o rollback → sicuri).
"""

import re
from pathlib import Path
from typing import Tuple, Optional, List
from core.logger import get_logger
from core.config_loader import ConfigLoader

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
ALLOWED_SOURCE_EXTENSIONS = {".py", ".json", ".yaml", ".yml", ".md", ".sh", ".txt"}


class SafetyValidator:
    def __init__(self, config: ConfigLoader) -> None:
        self.config = config
        self.enabled          = config.get("safety.enabled", True)
        self.whitelist_mode   = config.get("safety.whitelist_mode", True)
        self.allowed_apps     = set(app.lower() for app in config.get("safety.allowed_apps", []))
        self.blocked_commands = [cmd.lower() for cmd in config.get("safety.blocked_commands", [])]
        self.allowed_extensions = set(config.get("safety.allowed_file_extensions", []))
        self.restricted_paths   = config.get("safety.restricted_paths", [])
        self.max_command_length = config.get("safety.max_command_length", 500)

    def validate_action(self, action: dict) -> Tuple[bool, str]:
        if not self.enabled:
            return True, ""

        action_type = action.get("type", "").lower()

        validators = {
            "open_app":        self._validate_open_app,
            "write_file":      self._validate_file_write,
            "read_file":       self._validate_file_read,
            "run_script":      self._validate_run_script,
            "system_command":  self._validate_system_command,
            "delete_file":     self._validate_file_delete,
            # ── Nuovi tool self-improvement ──
            "edit_source":     self._validate_edit_source,
            "read_source":     lambda a: (True, ""),   # sempre consentito
            "list_source":     lambda a: (True, ""),
            "rollback_source": lambda a: (True, ""),
        }

        validator = validators.get(action_type)
        if validator:
            return validator(action)

        return True, ""  # tool non noti → gestiti dal registry

    # ------------------------------------------------------------------
    # Validator originali (invariati)
    # ------------------------------------------------------------------

    def _validate_open_app(self, action: dict) -> Tuple[bool, str]:
        target = action.get("target", "").lower().strip()
        if not target:
            return False, "Nessuna applicazione specificata"
        if len(target) > 100:
            return False, "Nome applicazione troppo lungo"
        if any(c in target for c in [".", "/", "\\", ";", "&", "|", "`", "$", ">"]):
            return False, f"Caratteri non validi nel nome applicazione: '{target}'"
        if self.whitelist_mode and self.allowed_apps:
            if not any(allowed in target for allowed in self.allowed_apps):
                return False, (
                    f"Applicazione '{target}' non nella whitelist. "
                    f"Consentite: {', '.join(sorted(self.allowed_apps))}"
                )
        return True, ""

    def _validate_file_write(self, action: dict) -> Tuple[bool, str]:
        path = action.get("path", "")
        for restricted in self.restricted_paths:
            if path.startswith(restricted):
                return False, f"Percorso ristretto: {restricted}"
        if ".." in path:
            return False, "Path traversal non consentito"
        import os
        ext = os.path.splitext(path)[1].lower()
        if self.allowed_extensions and ext not in self.allowed_extensions:
            return False, f"Estensione '{ext}' non consentita"
        return True, ""

    def _validate_file_read(self, action: dict) -> Tuple[bool, str]:
        path = action.get("path", "")
        for restricted in self.restricted_paths:
            if path.startswith(restricted):
                return False, f"Lettura da percorso ristretto non consentita: {restricted}"
        if ".." in path:
            return False, "Path traversal non consentito"
        return True, ""

    def _validate_run_script(self, action: dict) -> Tuple[bool, str]:
        script_name = action.get("script_name", "")
        if "/" in script_name or "\\" in script_name or ".." in script_name:
            return False, "Nome script non può contenere separatori di percorso"
        import os
        ext = os.path.splitext(script_name)[1].lower()
        if ext not in {".sh", ".py", ".bat"}:
            return False, f"Tipo script '{ext}' non consentito"
        return True, ""

    def _validate_system_command(self, action: dict) -> Tuple[bool, str]:
        command = action.get("command", "").lower()
        if len(command) > self.max_command_length:
            return False, f"Comando troppo lungo (max {self.max_command_length})"
        for blocked in self.blocked_commands:
            if blocked in command:
                return False, f"Pattern bloccato: '{blocked}'"
        if self.whitelist_mode:
            return False, (
                "Comandi di sistema diretti non consentiti in whitelist mode. "
                "Usa open_app o run_script."
            )
        return True, ""

    def _validate_file_delete(self, action: dict) -> Tuple[bool, str]:
        return False, "Cancellazione file non consentita per sicurezza."

    # ------------------------------------------------------------------
    # Nuovo validator: edit_source
    # ------------------------------------------------------------------

    def _validate_edit_source(self, action: dict) -> Tuple[bool, str]:
        """
        Valida l'operazione edit_source.
        Regole:
          - path deve essere relativo e dentro PROJECT_ROOT
          - nessun path traversal
          - estensione nella whitelist ALLOWED_SOURCE_EXTENSIONS
          - operation deve essere uno dei valori consentiti
          - new_text non deve contenere pattern di shell injection ovvi
        """
        rel_path  = action.get("path", "")
        operation = action.get("operation", "")
        new_text  = action.get("new_text", "")

        # Path traversal
        if ".." in rel_path:
            return False, "Path traversal non consentito in edit_source."

        # Estensione
        suffix = Path(rel_path).suffix.lower()
        if suffix not in ALLOWED_SOURCE_EXTENSIONS:
            return False, f"Estensione '{suffix}' non consentita per edit_source."

        # Il file deve essere dentro PROJECT_ROOT
        target = (PROJECT_ROOT / rel_path).resolve()
        if not str(target).startswith(str(PROJECT_ROOT)):
            return False, "Il file deve essere dentro la directory del progetto."

        # Operazione valida
        allowed_ops = {"replace", "replace_all", "append", "insert_after"}
        if operation not in allowed_ops:
            return False, f"Operazione '{operation}' non valida. Usa: {', '.join(allowed_ops)}"

        # Controlla new_text per pattern pericolosi ovvi (shell injection nel codice)
        danger_patterns = [
            r"subprocess\.call\(.*(rm|del|format|fdisk)",
            r"os\.system\(.*(rm -rf|deltree|shutdown)",
            r"__import__\(['\"]os['\"]\)\.system",
        ]
        for p in danger_patterns:
            if re.search(p, new_text, re.IGNORECASE):
                logger.warning(f"edit_source bloccato per pattern pericoloso: {p}")
                return False, f"Il testo proposto contiene pattern potenzialmente pericoloso."

        return True, ""

    # ------------------------------------------------------------------
    # Utilità
    # ------------------------------------------------------------------

    def validate_llm_output(self, output: str) -> Tuple[bool, str]:
        if not output:
            return True, ""
        injection_patterns = [
            r"ignore previous instructions",
            r"disregard all previous",
            r"you are now",
            r"new instructions:",
            r"system override",
            r"jailbreak",
        ]
        output_lower = output.lower()
        for pattern in injection_patterns:
            if re.search(pattern, output_lower):
                logger.warning(f"Possibile prompt injection: {pattern}")
                return False, f"Pattern sospetto rilevato: '{pattern}'"
        return True, ""

    def sanitize_string(self, value: str) -> str:
        dangerous = set(";&|`$><\\")
        return "".join(c for c in value if c not in dangerous)