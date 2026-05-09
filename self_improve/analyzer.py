"""
Self-Improvement Analyzer - Analyzes interaction logs and generates improvement proposals.
Uses the LLM to identify patterns and suggest system improvements.
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from self_improve.logger import InteractionLogger
from core.logger import get_logger
from core.config_loader import ConfigLoader

logger = get_logger(__name__)


ANALYSIS_PROMPT = """You are the self-improvement engine for JARVIS, an AI assistant.

Analyze the following interaction logs and identify:
1. Repeated failures or unrecognized commands
2. Inefficient action patterns
3. Missing tools that would be useful
4. Prompt improvements needed
5. Memory strategy improvements

INTERACTION STATISTICS:
{stats}

RECENT FAILURES:
{failures}

MOST COMMON ACTIONS:
{action_distribution}

Based on this analysis, generate improvement proposals as a JSON array.
Each proposal MUST follow this exact schema:

[
  {{
    "id": "patch_<timestamp>_<n>",
    "type": "prompt_improvement|new_tool|config_change|memory_improvement",
    "priority": "high|medium|low",
    "description": "Clear description of the improvement",
    "problem": "What problem this solves",
    "solution": "How to implement it",
    "data": {{
      // Type-specific data:
      // For prompt_improvement: {{"prompt_section": "...", "current": "...", "proposed": "..."}}
      // For new_tool: {{"tool_name": "...", "description": "...", "action_type": "..."}}
      // For config_change: {{"key": "...", "current_value": "...", "proposed_value": "..."}}
      // For memory_improvement: {{"strategy": "...", "rationale": "..."}}
    }}
  }}
]

Respond ONLY with the JSON array. No other text. Generate 1-5 proposals maximum.
If no improvements are needed, return an empty array: []
"""


class SelfImprovementAnalyzer:
    """
    Analyzes JARVIS's own interaction logs to identify improvement opportunities.
    Uses the LLM to generate structured improvement proposals.
    """

    def __init__(
        self,
        interaction_logger: InteractionLogger,
        llm,  # OllamaClient
        config: ConfigLoader
    ) -> None:
        self.interaction_logger = interaction_logger
        self.llm = llm
        self.config = config

    async def analyze(self) -> List[Dict[str, Any]]:
        """
        Run a full analysis cycle and return improvement proposals.
        """
        logger.info("Starting self-improvement analysis...")

        # 1. Gather data
        stats = await self.interaction_logger.get_stats()
        total = stats.get("total", 0)

        min_interactions = self.config.get(
            "self_improve.min_interactions_before_analysis", 10
        )

        if total < min_interactions:
            logger.info(
                f"Not enough interactions for analysis "
                f"({total}/{min_interactions} minimum)"
            )
            return []

        failures = await self.interaction_logger.get_failures(limit=20)
        action_dist = stats.get("action_distribution", {})

        # 2. Format for LLM
        prompt = ANALYSIS_PROMPT.format(
            stats=json.dumps(stats, indent=2),
            failures=json.dumps(failures[:10], indent=2),
            action_distribution=json.dumps(action_dist, indent=2)
        )

        # 3. Get LLM analysis
        logger.info("Querying LLM for improvement proposals...")
        try:
            response = await self.llm.generate(prompt, temperature=0.3, max_tokens=2000)

            if not response:
                logger.warning("Empty response from LLM during analysis")
                return []

            # 4. Parse proposals
            proposals = self._parse_proposals(response)
            logger.info(f"Analysis complete: {len(proposals)} proposals generated")
            return proposals

        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            return self._generate_rule_based_proposals(stats, failures)

    def _parse_proposals(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response into structured proposals."""
        response = response.strip()

        # Remove markdown code blocks if present
        if response.startswith("```"):
            lines = response.split("\n")
            response = "\n".join(
                line for line in lines
                if not line.startswith("```")
            )

        try:
            proposals = json.loads(response)
            if not isinstance(proposals, list):
                logger.warning("LLM returned non-list proposals")
                return []

            # Validate and add IDs if missing
            timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            valid = []
            for i, p in enumerate(proposals):
                if not isinstance(p, dict):
                    continue
                if "type" not in p or "description" not in p:
                    continue

                # Ensure ID
                if "id" not in p:
                    p["id"] = f"patch_{timestamp}_{i+1}"

                # Ensure required fields
                p.setdefault("priority", "medium")
                p.setdefault("problem", "Not specified")
                p.setdefault("solution", p.get("description", ""))
                p.setdefault("data", {})
                p["proposed_at"] = datetime.utcnow().isoformat()
                p["status"] = "pending"

                valid.append(p)

            return valid[:self.config.get("self_improve.max_patches_per_cycle", 5)]

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse proposals JSON: {e}")
            return []

    def _generate_rule_based_proposals(
        self,
        stats: Dict,
        failures: List[Dict]
    ) -> List[Dict[str, Any]]:
        """
        Generate simple rule-based proposals when LLM analysis fails.
        This ensures we always get some improvement even without LLM.
        """
        proposals = []
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")

        failure_rate = stats.get("failure_rate", 0)
        if failure_rate > 0.2:  # >20% failure rate
            proposals.append({
                "id": f"patch_{timestamp}_1",
                "type": "prompt_improvement",
                "priority": "high",
                "description": "High failure rate detected - improve action parsing",
                "problem": f"Failure rate is {failure_rate:.1%}, which is too high",
                "solution": "Clarify action format in system prompt and add examples",
                "data": {
                    "prompt_section": "RESPONSE FORMAT",
                    "metric": f"failure_rate={failure_rate:.1%}"
                },
                "proposed_at": datetime.utcnow().isoformat(),
                "status": "pending"
            })

        # Check for unrecognized action types in failures
        unknown_types = set()
        for f in failures:
            response = f.get("llm_response", "")
            if "Unknown action" in response:
                import re
                matches = re.findall(r"Unknown action '(\w+)'", response)
                unknown_types.update(matches)

        for unknown_type in list(unknown_types)[:2]:
            proposals.append({
                "id": f"patch_{timestamp}_{len(proposals)+1}",
                "type": "new_tool",
                "priority": "medium",
                "description": f"LLM requested unknown action type '{unknown_type}'",
                "problem": f"No tool found for action type: {unknown_type}",
                "solution": f"Create a new tool module for '{unknown_type}' or add to system prompt",
                "data": {
                    "action_type": unknown_type,
                    "occurrences": sum(
                        1 for f in failures
                        if f"Unknown action '{unknown_type}'" in f.get("llm_response", "")
                    )
                },
                "proposed_at": datetime.utcnow().isoformat(),
                "status": "pending"
            })

        return proposals

    async def get_performance_report(self) -> str:
        """Generate a human-readable performance report."""
        stats = await self.interaction_logger.get_stats()
        failures = await self.interaction_logger.get_failures(limit=5)

        report = [
            "=== JARVIS Self-Improvement Report ===",
            f"Total interactions: {stats.get('total', 0)}",
            f"Failure rate: {stats.get('failure_rate', 0):.1%}",
            f"",
            "Most used actions:",
        ]

        for action, count in sorted(
            stats.get("action_distribution", {}).items(),
            key=lambda x: -x[1]
        )[:5]:
            report.append(f"  - {action}: {count} times")

        if failures:
            report.append("\nRecent failures:")
            for f in failures[:3]:
                report.append(
                    f"  - [{f.get('timestamp', '?')[:16]}] "
                    f"{f.get('user_input', '')[:80]}"
                )

        return "\n".join(report)