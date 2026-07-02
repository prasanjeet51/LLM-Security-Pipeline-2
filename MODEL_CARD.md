# Model Card — Hybrid LLM Jailbreak + Prompt Injection Detector

## Model Details

- **Architecture**: Hybrid pipeline. Stage A is `answerdotai/ModernBERT-base`
  fine-tuned with LoRA adapters as a 3-class sequence classifier. Stage B is
  `meta-llama/Llama-Guard-3-8B`, invoked only on uncertain or risky inputs. A
  GPT-2 perplexity gate and a FAISS similarity gate run in front of Stage A on
  every request.
- **Classes**: `safe = 0`, `jailbreak = 1`, `indirect_injection = 2`.
- **Inference runtimes**: PyTorch (default) or ONNX Runtime (2–5× speedup, with
  numerical-tolerance check 1e-4 against PyTorch and graceful fallback).
- **Decision layer**: a deterministic policy gate maps classifier output +
  perplexity + similarity + source-type + confidence into one of `allow`,
  `block`, `human_review`. The model can be wrong; the policy gate decides the
  outcome.
- **Authoring**: built and trained by me as a portfolio project. I am the sole
  author of the code, training pipeline, evaluation, and red-team harness.

## Intended Use

- Input safety filtering for LLM-powered applications (chatbots, agents,
  retrieval-augmented systems).
- Preflight check on retrieved content (documents, search results, tool
  outputs) for indirect prompt injection.
- Educational / portfolio reference for a defense-in-depth detection stack.

**This is not a sole safety control.** It must be deployed as part of a
defense-in-depth strategy alongside output filtering, sandboxing of tool calls,
human review for high-impact actions, and a monitoring layer that watches for
drift and abuse patterns.

## Training Data

| Source | Hub / location | Role |
|---|---|---|
| JailbreakBench | Hugging Face | jailbreak prompts |
| AdvBench | local CSV `data/raw/harmful_behaviors.csv` | jailbreak prompts |
| WildJailbreak | Hugging Face (streaming, capped) | jailbreak prompts |
| deepset/prompt-injections | Hugging Face | indirect injection |
| HackAPrompt | Hugging Face (streaming, BIPIA fallback) | injection / jailbreak |
| C4 / OpenWebText safe corpus | Hugging Face | safe class |

- All sources are single-turn. Every row carries `is_multiturn = False`.
- Unified schema: `sample_id, text, label, source_dataset, source_type,
  language, is_multiturn`.
- Actual training distribution (per-class counts and split sizes) is recorded
  in `reports/results.json` under the `training_samples` key after each training
  run.

## Evaluation Results

Pulled from `reports/results.json` (run dated 2026-04-17):

| Model | Accuracy | Weighted F1 | Jailbreak Recall | Indirect Recall | FPR (Safe) | Latency p50 | Latency p95 |
|---|---|---|---|---|---|---|---|
| TF-IDF + LinearSVC (baseline) | 0.9222 | 0.9217 | 0.4921 | 0.5472 | 0.024 | 4.14 ms | 5.62 ms |
| Stage A ModernBERT + LoRA (hybrid) | **0.9988** | **0.9988** | **0.9947** | **0.9906** | **0.0004** | 206.39 ms | 293.7 ms |

Reliability diagrams (calibration curves) and confusion matrices for the
hybrid model are in `reports/figures/`.

### Red-team evaluation

From `reports/redteam/attack_run_20260418.json`:

- Single-turn ASR: **0.226** (driven by back-translation; see Known Limitations).
- Multi-turn ASR: **0.056**.
- Block rates: `direct_jailbreak = 1.00`, `indirect_injection = 1.00`,
  `multilingual = 1.00`, `typoglycemia = 1.00`, `multi_turn_crescendo = 1.00`,
  `goal_hijacking = 0.875`, `back_translation = 0.53`.
- One critical finding under `goal_hijacking` (no session memory by design — see
  Known Limitations #3 and #4).

## Stage B (Llama Guard 3) — Hardware Requirements

Stage B is fully implemented but disabled by default in the shipped config
because most local development machines cannot host the 8B-parameter judge.
Enabling it requires:

- **GPU**: 16 GB VRAM minimum — A100, H100, or 2× RTX 3090 / 4090 with
  `device_map="auto"`. CPU inference is not supported in any practical sense.
- **Hugging Face access**: gated model. Request access at
  https://huggingface.co/meta-llama/Llama-Guard-3-8B and authenticate with
  `huggingface-cli login` before first run.
- **Config flag**: set `model.stage_b.enabled: true` in
  `config/config.yaml`.

Stage B is invoked only when the policy gate decides escalation is warranted:
uncertain Stage A confidence, multi-turn conversation history, presence of
external context (`source_type` other than `user_input`), or text the
normalizer flagged as obfuscated. Local development uses Stage A only;
production deployments on GPU enable the full pipeline. The Hugging Face Space
demo proxies Stage B through hosted Llama Guard providers (Together AI / Groq)
to avoid running the 8B model in the Space itself.

## Known Limitations

1. **Adaptive multi-turn attacks from large reasoning models are out of scope.**
   The red-team harness uses static mutation operators (back-translation,
   typoglycemia, multilingual, crescendo, etc.). LRM-generated adaptive
   multi-turn attacks (Hagendorff et al., *Nature Communications* 2026) are not
   covered and represent the current state-of-the-art threat vector. A
   detector trained against static mutations will look strong on this kind of
   harness and weaker against an attacker that adapts in the loop.

2. **Perplexity gate is tuned on English text.** Non-English inputs and
   code-heavy inputs naturally have high perplexity under a GPT-2 model trained
   primarily on English web text. This produces false positives on legitimate
   non-English queries and code-mode prompts. Monitor false-positive rate per
   `source_type` and per detected language; consider per-language perplexity
   thresholds before enabling the gate as a hard block.

3. **Stage A is trained on single-turn data only.** Multi-turn handling at
   inference is a concatenation heuristic (history + current turn,
   classified together), not learned multi-turn behavior. Crescendo-style
   attacks that rely on incremental drift across many turns are partially
   addressed by concatenation but are not the model's strength.

4. **Not a sole safety control.** This system catches input-side attacks. It
   does not catch model-side failure modes (hallucination, harmful content
   generated without an attack), tool-call abuse, or downstream agent
   misbehavior. Never deploy as the only guardrail. Pair with output filtering,
   tool-call review, and runtime monitoring.

5. **Back-translation bypass.** 35 of 75 back-translated attacks evaded Stage A
   in the most recent red-team run (`single_turn_asr = 0.226` on this
   operator). Back-translated text is clean English with a different surface
   form, so the FAISS similarity gate does not catch it and Stage A's pattern
   features generalize less well than to direct jailbreaks. Mitigations:
   (a) add back-translated variants of known attacks to the FAISS index, and
   (b) lower the similarity threshold from 0.85 to 0.80 in
   `config/config.yaml`. Both should be re-evaluated against the red-team
   harness before shipping.

## EU AI Act Considerations

This model card is part of the project's transparency posture. It does not
constitute legal advice; deployers remain responsible for their own
compliance assessment.

- **Article 6 — High-risk classification.** If used in safety-critical
  contexts (e.g., a layer in front of a decision-support system in healthcare,
  finance, or critical infrastructure), this detector could fall within the
  scope of "high-risk AI". Deployers in those contexts should confirm
  classification before integration.
- **Article 9 — Risk management.** The layered defense (normalize →
  perplexity → similarity → Stage A → policy gate → optional Stage B) is the
  risk-mitigation strategy. Each stage is independently testable, and the
  red-team harness exercises the full pipeline against named threat
  categories.
- **Article 11 — Technical documentation.** This MODEL_CARD, together with
  README.md and the contents of `reports/`, is intended to satisfy the
  technical-documentation requirement for the detector itself. Datasets,
  training code, evaluation metrics, calibration plots, and red-team results
  are all reproducible from the repository.
- **Article 14 — Human oversight.** The `human_review` decision path exists
  precisely so that uncertain inputs are routed to a human, not silently
  allowed or silently blocked. Deployers must wire `human_review` into an
  actual review queue; the detector cannot satisfy Article 14 on its own.
- **Transparency.** End users of any application that integrates this detector
  should be informed that their input is being classified for safety, in line
  with the broader transparency obligations of the Act.
