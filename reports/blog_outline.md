# Blog Outline

**Title:** Building a production-grade LLM safety layer from scratch: architecture, training, and hard lessons

**Platform:** Medium or Substack
**Estimated reading time:** 8–10 minutes
**Tone:** Technical but accessible. Builder sharing real experience, not marketing. Include failures.

---

## Section 1 — The problem (600 words)

**Hook:** Most LLM apps treat user input like a trusted function argument. It isn't.

**Cover:**
- What a direct jailbreak actually looks like (show a real example)
- What indirect prompt injection looks like — and why it's harder (the malicious PDF that executes when your model summarizes it)
- Why output filters don't help: by the time you filter the response, the model has already followed the injected instruction
- The 2024 OWASP LLM Top 10: prompt injection is #1. Quote the number.

**End:** "I wanted to know if you could build a layered, deterministic defense. Here's what I found."

---

## Section 2 — Why a hybrid approach (400 words)

**Cover:**
- A single classifier has a fixed precision/recall tradeoff. The jailbreak attacker only needs to find the cheap inputs that slip through.
- Regex/keyword filters: break the moment the attacker reads the source code.
- The insight: cheap filters in front (perplexity, FAISS), expensive classifier in the middle, deterministic gate at the end.
- Why the policy gate is the last word — not the model. Models are probabilistic. Gates are not.

**Key idea to land:** "Defense in depth" is a physical security concept. It works for LLMs for the same reason it works for buildings.

---

## Section 3 — Architecture deep-dive (700 words)

**Walk through each of the 5 layers:**

1. **Normalizer** — why homoglyphs and zero-width chars are the oldest evasion trick in the book; what normalization actually does
2. **Perplexity gate (GPT-2)** — the intuition: adversarially-optimized suffixes look like noise to a language model. Show a GCG suffix example. Explain the threshold and why 500 is a reasonable default.
3. **FAISS similarity** — why nearest-neighbor lookup over a known-attack index gives near-zero false negatives for variants of seen attacks. How it fails (novel attacks it hasn't seen).
4. **Stage A (ModernBERT-base + LoRA)** — why LoRA over full fine-tuning (parameter efficiency at 8192-token context), why ModernBERT over BERT/RoBERTa (longer context, more recent pretraining), why 3 classes (safe/jailbreak/indirect_injection — they're different attack surfaces).
5. **Policy gate** — show the decision table. Explain why this is deterministic by design: you tune thresholds; a model doesn't decide policy.

**Include:** the ASCII pipeline diagram from the README.

---

## Section 4 — The data problem (500 words)

**Cover:**
- What training data exists: AdvBench, JailbreakBench, HarmBench, lmsys-chat-1m, wildguard — and which ones are gated behind academic access requests.
- What we used: 5 open datasets, unified schema (sample_id, text, label, source_dataset, source_type, language, is_multiturn). ~50k rows after deduplication.
- The class imbalance problem: indirect_injection is rare. Show the class distribution. Explain oversampling strategy.
- What's missing: genuine multi-turn attack sequences. Every row is `is_multiturn=False`. This is a limitation, and it shows in the red-team results.

**Honest note:** "If you know of a good open multi-turn jailbreak dataset, please open an issue."

---

## Section 5 — Training on Kaggle (500 words)

**Cover:**
- Why Kaggle: free P100 GPU, reproducible kernel, no cloud cost
- The OOD bug we found: a data leakage issue where the evaluation set contained prompts similar to training prompts. How we found it (evaluating by source_dataset, not just overall). How we fixed it (hard source-stratified split).
- LoRA training details: rank=16, alpha=32, target_modules=["query", "value"], 3 epochs, ~45 minutes on P100
- ONNX export + INT8 quantization: 1.78x CPU speedup (334ms → 188ms, 100 runs), verified within 1e-4 tolerance

**Include:** training curve screenshot (loss/f1 vs epoch)

---

## Section 6 — Evaluation numbers (400 words)

**Cover:**
- The benchmark table (reproduce from README)
- What FPR 0.04% means in practice: 1 false flag per 2,500 legitimate requests. At 10,000 requests/day, that's 4 unhappy users per day. At 1,000 requests/day, it's ~3 per week. Is that acceptable? Depends on your application.
- The red-team table — walk through back-translation honestly
- Single-turn ASR 22.6% is almost entirely back-translation. Multi-turn ASR 5.6%.
- Calibration: reliability diagram. Temperature scaling brings ECE from 0.08 to 0.02.

---

## Section 7 — What I'd do differently (400 words)

**Cover:**
- Stage B (Llama Guard 3 8B) requires a GPU I don't have locally. In production you'd run it on a dedicated inference server. I'd design the architecture to make Stage B a remote call from the start, not a local import.
- Multi-turn training data: the single-turn limitation is real. A crescendo attack that slowly escalates across 10 messages will look safe on any individual message.
- Perplexity gate threshold: 500 is a reasonable default but it's dataset-dependent. I'd ship a calibration notebook instead of a hardcoded default.
- Privacy audit: I ran it, found nothing. But I should have set up detect-secrets on Day 0, not Day 13.

---

## Section 8 — How to use it in your project (300 words)

**The three use cases:**

1. **Before your LLM call** (most common):
```python
from jailbreak_detector import detect

result = detect(user_prompt)
if result.blocked:
    return {"error": "Input rejected", "reason": result.reason}
response = llm.complete(user_prompt)
```

2. **RAG pipeline** (indirect injection):
```python
result = detect(user_prompt, context=retrieved_doc)
if result.blocked:
    return {"error": "Retrieved content contains injection attempt"}
```

3. **Multi-turn chat**:
```python
result = detect(user_message, history=conversation_history)
```

**Close:** Link to GitHub, HF Space, and offer to answer questions in the comments.

---

## Notes for writing the full post

- Each section maps to roughly one day of the 15-day build.
- Include real code snippets, not pseudocode.
- The red-team section is where most readers will learn something new — spend extra time there.
- Don't editorialize about LLM safety in general. Stick to what you built and what you measured.
- Add a "Further reading" section at the end: Perez & Ribeiro (2022) prompt injection paper, GCG paper (Zou et al. 2023), OWASP LLM Top 10.
