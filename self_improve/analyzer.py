"""
Self-Improvement Engine v2 — Analisi log + patch automatiche
--------------------------------------------------------------
Miglioramenti rispetto all'originale:
  - Analisi statistica REALE dei log (pattern errori, action type sconosciuti,
    latenze, tasso di successo per tipo di azione).
  - Genera patch CONFIG concrete applicate subito (senza approvazione manuale
    per i parametri non critici).
  - Regole euristiche veloci che non richiedono LLM (fallback).
  - Report human-readable periodico scritto in data/self_improve_report.md.
  - Nuovo metodo: learn_from_interaction() chiamato dopo ogni risposta →
    aggiorna statistiche in tempo reale senza aspettare il ciclo completo.
"""

import json
import re
import asyncio
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from self_improve.logger import InteractionLogger
from core.logger import get_logger
from core.config_loader import ConfigLoader

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
REPORT_PATH  = PROJECT_ROOT / "data" / "self_improve_report.md"


# ---------------------------------------------------------------------------
# Metriche in memoria (reset ad ogni avvio)
# ---------------------------------------------------------------------------

class LiveMetrics:
    """Metriche raccolte in tempo reale durante la sessione corrente."""

    def __init__(self):
        self.total          = 0
        self.successes      = 0
        self.failures       = 0
        self.latencies      = []          # secondi
        self.action_counts  = Counter()   # type → count
        self.error_messages = []          # ultimi 50 errori
        self.unknown_actions= Counter()   # action type non riconosciuto → count
        self.session_start  = datetime.utcnow()

    def record(self, record: Dict):
        self.total += 1
        success = record.get("success", True)
        if success:
            self.successes += 1
        else:
            self.failures += 1
            err = record.get("error") or record.get("final_response", "")
            if err:
                self.error_messages = (self.error_messages + [str(err)])[-50:]

        # Latenza
        latency = record.get("latency_seconds")
        if latency:
            self.latencies.append(float(latency))

        # Tipo azione
        action = record.get("parsed_action")
        if isinstance(action, dict):
            atype = action.get("type", "unknown")
            self.action_counts[atype] += 1
            # Controlla se era un'azione sconosciuta
            final = record.get("final_response", "")
            if "Unknown action" in final or "unknown action" in final.lower():
                self.unknown_actions[atype] += 1

    @property
    def failure_rate(self) -> float:
        return self.failures / self.total if self.total > 0 else 0.0

    @property
    def avg_latency(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0

    def summary(self) -> Dict:
        return {
            "total": self.total,
            "successes": self.successes,
            "failures": self.failures,
            "failure_rate": round(self.failure_rate, 3),
            "avg_latency_s": round(self.avg_latency, 2),
            "top_actions": self.action_counts.most_common(10),
            "unknown_actions": dict(self.unknown_actions),
            "session_minutes": round(
                (datetime.utcnow() - self.session_start).total_seconds() / 60, 1
            ),
        }


_live_metrics = LiveMetrics()


# ---------------------------------------------------------------------------
# Analisi log storica
# ---------------------------------------------------------------------------

async def _load_recent_records(
    interaction_logger: InteractionLogger,
    n: int = 500
) -> List[Dict]:
    """Carica gli ultimi N record dal file JSONL."""
    return await interaction_logger.read_recent(n)


def _analyse_records(records: List[Dict]) -> Dict:
    """Analisi statistica completa dei record storici."""
    if not records:
        return {}

    total    = len(records)
    failures = [r for r in records if not r.get("success", True)]
    actions  = Counter()
    unknown  = Counter()
    latencies= []

    for r in records:
        a = r.get("parsed_action")
        if isinstance(a, dict):
            t = a.get("type", "?")
            actions[t] += 1
        lat = r.get("latency_seconds")
        if lat:
            latencies.append(float(lat))
        final = r.get("final_response", "")
        for m in re.findall(r"Unknown action '(\w+)'", final):
            unknown[m] += 1

    # Errori più frequenti (prime 80 char)
    error_snippets = Counter()
    for r in failures:
        msg = (r.get("error") or r.get("final_response", ""))[:80].strip()
        if msg:
            error_snippets[msg] += 1

    return {
        "total": total,
        "failure_rate": len(failures) / total,
        "avg_latency_s": sum(latencies) / len(latencies) if latencies else 0,
        "top_actions": actions.most_common(10),
        "unknown_actions": dict(unknown),
        "top_errors": error_snippets.most_common(5),
        "date_range": {
            "first": records[0].get("timestamp", "?"),
            "last":  records[-1].get("timestamp", "?"),
        }
    }


# ---------------------------------------------------------------------------
# Generatore di patch euristiche (no LLM necessario)
# ---------------------------------------------------------------------------

def _generate_heuristic_patches(
    stats: Dict,
    config: ConfigLoader
) -> List[Dict]:
    """
    Genera patch concrete basandosi su soglie statistiche, senza LLM.
    Tutte le patch di tipo 'config_change' vengono applicate subito.
    """
    patches = []
    ts      = datetime.utcnow().strftime("%Y%m%d%H%M%S")

    # ── Latenza alta → suggerisci modello più veloce ─────────────────────
    avg_lat = stats.get("avg_latency_s", 0)
    if avg_lat > 8.0:
        current_tokens = config.get("llm.max_tokens", 768)
        new_tokens     = max(512, int(current_tokens * 0.8))
        patches.append({
            "id": f"patch_{ts}_latency",
            "type": "config_change",
            "priority": "high",
            "description": f"Riduzione max_tokens per migliorare latenza (avg: {avg_lat:.1f}s)",
            "auto_apply": True,
            "data": {
                "key": "llm.max_tokens",
                "current_value": current_tokens,
                "proposed_value": new_tokens,
            }
        })

    # ── Tasso fallimento alto → più history messages ──────────────────────
    fail_rate = stats.get("failure_rate", 0)
    if fail_rate > 0.25:
        cur_hist = config.get("llm.history_messages", 6)
        if cur_hist < 10:
            patches.append({
                "id": f"patch_{ts}_history",
                "type": "config_change",
                "priority": "medium",
                "description": f"Aumento history_messages (failure rate: {fail_rate:.0%})",
                "auto_apply": True,
                "data": {
                    "key": "llm.history_messages",
                    "current_value": cur_hist,
                    "proposed_value": min(cur_hist + 2, 12),
                }
            })

    # ── Azioni sconosciute → nota informativa ────────────────────────────
    for action_type, count in stats.get("unknown_actions", {}).items():
        if count >= 2:
            patches.append({
                "id": f"patch_{ts}_tool_{action_type}",
                "type": "missing_tool_note",
                "priority": "medium",
                "description": f"Tool '{action_type}' richiesto {count}× ma non trovato",
                "auto_apply": False,
                "data": {"action_type": action_type, "occurrences": count},
            })

    # ── Timeout ripetuti → aumenta read_timeout ──────────────────────────
    timeout_errors = sum(
        1 for err, _ in stats.get("top_errors", [])
        if "timeout" in err.lower() or "timed out" in err.lower()
    )
    if timeout_errors >= 2:
        cur_timeout = config.get("llm.read_timeout", 300)
        patches.append({
            "id": f"patch_{ts}_timeout",
            "type": "config_change",
            "priority": "medium",
            "description": f"Aumento read_timeout ({timeout_errors} timeout rilevati)",
            "auto_apply": True,
            "data": {
                "key": "llm.read_timeout",
                "current_value": cur_timeout,
                "proposed_value": min(cur_timeout + 60, 600),
            }
        })

    return patches


# ---------------------------------------------------------------------------
# Applicazione automatica patch config
# ---------------------------------------------------------------------------

def _apply_config_patches(patches: List[Dict], config: ConfigLoader) -> List[str]:
    """Applica subito le patch di tipo config_change marcate auto_apply."""
    applied = []
    config_path = PROJECT_ROOT / "config" / "config.json"

    try:
        raw = json.loads(config_path.read_text())
    except Exception as e:
        logger.error(f"Impossibile leggere config per patch: {e}")
        return []

    changed = False
    for patch in patches:
        if not patch.get("auto_apply"):
            continue
        if patch.get("type") != "config_change":
            continue

        key   = patch["data"].get("key", "")
        value = patch["data"].get("proposed_value")
        if not key or value is None:
            continue

        # Naviga il dict con dot-notation
        parts   = key.split(".")
        current = raw
        try:
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            old = current.get(parts[-1])
            current[parts[-1]] = value
            config.set(key, value)  # Override runtime immediato
            applied.append(f"{key}: {old} → {value}")
            changed = True
            logger.info(f"Self-improve patch applicata: {key} = {value} (era {old})")
        except Exception as e:
            logger.warning(f"Patch {key} fallita: {e}")

    if changed:
        try:
            config_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False))
        except Exception as e:
            logger.error(f"Salvataggio config patch fallito: {e}")

    return applied


# ---------------------------------------------------------------------------
# Report Markdown
# ---------------------------------------------------------------------------

def _write_report(stats: Dict, patches: List[Dict], applied: List[str]) -> None:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# JARVIS Self-Improvement Report",
        f"*{ts}*",
        "",
        "## Statistiche",
        f"- Interazioni totali: {stats.get('total', 0)}",
        f"- Tasso fallimento: {stats.get('failure_rate', 0):.1%}",
        f"- Latenza media: {stats.get('avg_latency_s', 0):.1f}s",
        "",
        "## Azioni più usate",
    ]
    for action, count in stats.get("top_actions", []):
        lines.append(f"- `{action}`: {count}×")

    if stats.get("unknown_actions"):
        lines += ["", "## Azioni non riconosciute"]
        for a, c in stats["unknown_actions"].items():
            lines.append(f"- `{a}`: {c}×")

    if stats.get("top_errors"):
        lines += ["", "## Errori più frequenti"]
        for err, cnt in stats["top_errors"]:
            lines.append(f"- ({cnt}×) `{err[:100]}`")

    lines += ["", "## Patch generate"]
    for p in patches:
        auto = "✅ auto-applicata" if p.get("auto_apply") else "⏳ manuale"
        lines.append(f"- [{auto}] **{p['type']}** — {p['description']}")

    if applied:
        lines += ["", "## Modifiche applicate questa sessione"]
        for a in applied:
            lines.append(f"- {a}")

    try:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Report self-improvement scritto: {REPORT_PATH}")
    except Exception as e:
        logger.warning(f"Report write failed: {e}")


# ---------------------------------------------------------------------------
# Analizzatore principale (compatibile con orchestrator.py esistente)
# ---------------------------------------------------------------------------

class SelfImprovementAnalyzer:
    """
    Analizzatore v2: combina euristica veloce + LLM opzionale.
    Compatibile con l'interfaccia usata da orchestrator.py.
    """

    def __init__(
        self,
        interaction_logger: InteractionLogger,
        llm,
        config: ConfigLoader
    ) -> None:
        self.interaction_logger = interaction_logger
        self.llm    = llm
        self.config = config

    def record_interaction(self, record: Dict) -> None:
        """Chiamato dopo ogni interazione per aggiornare le metriche live."""
        _live_metrics.record(record)

    async def analyze(self) -> List[Dict[str, Any]]:
        """Ciclo completo: carica log, analizza, genera patch, applica automaticamente."""
        logger.info("Self-improvement: avvio analisi...")

        records = await _load_recent_records(self.interaction_logger, n=500)
        stats   = _analyse_records(records)

        if not stats or stats.get("total", 0) < self.config.get(
            "self_improve.min_interactions_before_analysis", 10
        ):
            return []

        # 1. Patch euristiche (veloci, no LLM)
        patches = _generate_heuristic_patches(stats, self.config)

        # 2. Patch LLM (opzionale, solo se disponibile)
        if self.config.get("self_improve.use_llm_for_analysis", False):
            try:
                llm_patches = await self._llm_analyze(stats)
                patches.extend(llm_patches)
            except Exception as e:
                logger.warning(f"Analisi LLM fallita, uso solo euristiche: {e}")

        # 3. Applica automaticamente le patch sicure
        applied = _apply_config_patches(patches, self.config)

        # 4. Scrivi report
        _write_report(stats, patches, applied)

        logger.info(
            f"Self-improvement: {len(patches)} patch generate, "
            f"{len(applied)} applicate automaticamente"
        )
        return patches

    async def _llm_analyze(self, stats: Dict) -> List[Dict]:
        """Analisi LLM (chiamata solo se llm.use_llm_for_analysis=true)."""
        prompt = (
            "Sei il motore di auto-miglioramento di JARVIS. "
            "Analizza queste statistiche e rispondi SOLO con un array JSON "
            "di al massimo 3 proposte di miglioramento.\n\n"
            f"STATISTICHE:\n{json.dumps(stats, indent=2, ensure_ascii=False)}\n\n"
            "Schema proposta: {\"type\": \"config_change|prompt_improvement|new_tool\", "
            "\"description\": \"...\", \"data\": {...}}\n"
            "Risposta SOLO JSON, nessun testo extra."
        )
        try:
            response = await self.llm.generate(prompt, temperature=0.2, max_tokens=800)
            response = response.strip().lstrip("```json").rstrip("```").strip()
            proposals = json.loads(response)
            if isinstance(proposals, list):
                return proposals[:3]
        except Exception as e:
            logger.debug(f"LLM analyze parse error: {e}")
        return []

    async def get_performance_report(self) -> str:
        """Report human-readable per l'utente (chiamato su richiesta)."""
        live = _live_metrics.summary()
        records = await _load_recent_records(self.interaction_logger, 200)
        hist = _analyse_records(records)

        lines = [
            "=== JARVIS Performance Report ===",
            f"Sessione corrente: {live['session_minutes']} minuti",
            f"  Interazioni: {live['total']} | Fallimenti: {live['failures']} ({live['failure_rate']:.0%})",
            f"  Latenza media: {live['avg_latency_s']:.1f}s",
        ]
        if hist:
            lines += [
                "",
                f"Storico ({hist.get('total',0)} interazioni):",
                f"  Tasso fallimento: {hist.get('failure_rate',0):.0%}",
                f"  Latenza media: {hist.get('avg_latency_s',0):.1f}s",
            ]
            for a, c in hist.get("top_actions", [])[:5]:
                lines.append(f"  - {a}: {c}×")

        if REPORT_PATH.exists():
            lines.append(f"\nReport dettagliato: {REPORT_PATH}")

        return "\n".join(lines)