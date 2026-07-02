"""Stage B: Llama Guard 3 escalation judge (runs only when escalation triggers)."""

import os
import time
from typing import Any, Optional

from src.logger import get_logger

# Llama Guard 3 18-category taxonomy (S1..S18).
LLAMA_GUARD_CATEGORIES: tuple[str, ...] = (
    "S1_violent_crimes",
    "S2_non_violent_crimes",
    "S3_sex_crimes",
    "S4_child_exploitation",
    "S5_defamation",
    "S6_specialized_advice",
    "S7_privacy",
    "S8_intellectual_property",
    "S9_indiscriminate_weapons",
    "S10_hate",
    "S11_self_harm",
    "S12_sexual_content",
    "S13_elections",
    "S14_code_interpreter_abuse",
    "S15_misinformation",
    "S16_prompt_injection",
    "S17_unsafe_tool_use",
    "S18_jailbreak",
)

_SAFE_STUB: dict[str, Any] = {
    "is_safe": True,
    "violation_categories": [],
    "risk_score": 0.0,
}

_RETRY_DELAYS: tuple[float, ...] = (1.0, 2.0, 4.0)


class StageBJudge:
    """Pluggable adapter around Llama Guard 3. Escalation-only — never batch."""

    def __init__(
        self,
        config: dict[str, Any],
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ) -> None:
        """Initialise from config; API/local mode controlled by stage_b.use_api."""
        stage_cfg: dict[str, Any] = config["model"]["stage_b"]
        self._model_name: str = str(stage_cfg["model_name"])
        self._enabled: bool = bool(stage_cfg.get("enabled", True))
        self._use_api: bool = bool(stage_cfg.get("use_api", False))
        self._api_key: str = os.environ.get("TOGETHER_API_KEY", "")
        self._model = model
        self._tokenizer = tokenizer
        self._logger = get_logger(__name__)

    def _ensure_loaded(self) -> None:
        """Download the gated Llama Guard 3 model on first use."""
        if self._model is not None and self._tokenizer is not None:
            return
        # pragma: no cover - heavy gated model path
        from transformers import (  # pragma: no cover
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)  # nosec B615
        self._model = AutoModelForCausalLM.from_pretrained(  # nosec B615
            self._model_name
        )

    def judge(self, text: str, stage_a_result: dict[str, Any]) -> dict[str, Any]:
        """Return a Llama-Guard-shaped verdict dict.

        Only called by HybridPipeline when PolicyGate.should_escalate() is True.
        """
        if not self._enabled:
            return dict(_SAFE_STUB)

        prompt = self._build_prompt(text)

        if self._use_api:
            return self._judge_via_api(prompt, stage_a_result)

        return self._judge_local(text, prompt, stage_a_result)

    def _judge_via_api(
        self, prompt: str, stage_a_result: dict[str, Any]
    ) -> dict[str, Any]:
        """Call Together AI with exponential backoff; fall back to safe_stub."""
        import together  # type: ignore[import]

        client = together.Together(api_key=self._api_key)
        last_exc: Exception = RuntimeError("no attempts made")

        for attempt, delay in enumerate(_RETRY_DELAYS, start=1):
            try:
                response = client.chat.completions.create(
                    model="meta-llama/Llama-Guard-3-8B",
                    messages=[{"role": "user", "content": prompt}],
                )
                choice_msg = response.choices[0].message
                msg_content = (
                    choice_msg.content if choice_msg is not None else ""
                ) or ""
                raw: str = str(msg_content).strip().lower()
                verdict = self._parse_response(raw)
                self._logger.info(
                    "stage_b_judged_api",
                    extra={
                        "is_safe": verdict["is_safe"],
                        "n_categories": len(verdict["violation_categories"]),
                        "stage_a_label": stage_a_result.get("label"),
                        "attempt": attempt,
                    },
                )
                return verdict
            except Exception as exc:
                last_exc = exc
                self._logger.warning(
                    "stage_b_api_retry",
                    extra={"attempt": attempt, "error": str(exc), "delay": delay},
                )
                if attempt < len(_RETRY_DELAYS):
                    time.sleep(delay)

        self._logger.error(
            "stage_b_api_failed",
            extra={"error": str(last_exc), "fallback": "safe_stub"},
        )
        return dict(_SAFE_STUB)

    def _judge_local(
        self, text: str, prompt: str, stage_a_result: dict[str, Any]
    ) -> dict[str, Any]:
        """Run Llama Guard 3 locally and return the parsed verdict dict."""
        self._ensure_loaded()
        assert self._tokenizer is not None
        assert self._model is not None

        import torch

        encoded = self._tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output_ids = self._model.generate(**encoded, max_new_tokens=64)
        response_text = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
        verdict = self._parse_response(response_text)
        self._logger.info(
            "stage_b_judged",
            extra={
                "is_safe": verdict["is_safe"],
                "n_categories": len(verdict["violation_categories"]),
                "stage_a_label": stage_a_result.get("label"),
            },
        )
        return verdict

    def _build_prompt(self, text: str) -> str:
        """Build the Llama Guard 3 safety-check prompt for text."""
        cats = "\n".join(f"- {c}" for c in LLAMA_GUARD_CATEGORIES)
        return (
            "Task: Check if the following user message is safe against the "
            "Llama Guard 3 taxonomy.\n"
            f"Categories:\n{cats}\n\n"
            f"User message: {text}\n\n"
            "Respond with 'safe' or 'unsafe' followed by matching category codes."
        )

    @staticmethod
    def _parse_response(response: str) -> dict[str, Any]:
        """Parse Llama Guard output into {is_safe, violation_categories, risk_score}."""
        lowered = response.lower()
        is_unsafe = "unsafe" in lowered
        is_safe = not is_unsafe

        matched: list[str] = []
        for cat in LLAMA_GUARD_CATEGORIES:
            code = cat.split("_", 1)[0]  # e.g. "S16"
            if code.lower() in lowered or cat.lower() in lowered:
                matched.append(cat)

        if is_safe:
            risk_score = 0.0
        elif matched:
            risk_score = 0.9
        else:
            risk_score = 0.6
        return {
            "is_safe": is_safe,
            "violation_categories": matched,
            "risk_score": risk_score,
        }
