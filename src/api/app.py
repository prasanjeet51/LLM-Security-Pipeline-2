"""FastAPI application — P1 Hybrid Jailbreak Detector API."""

import json
import os
import time
import uuid
from typing import Any, AsyncGenerator, Optional

import psutil
from fastapi import FastAPI, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sse_starlette.sse import EventSourceResponse

from src.api.feedback import FeedbackStore
from src.api.schemas import (
    BatchClassifyRequest,
    BatchClassifyResponse,
    ClassifyRequest,
    ClassifyResponse,
    FeedbackRequest,
    HealthResponse,
)
from src.config import load_config
from src.exceptions import ProjectBaseError
from src.hybrid.pipeline import HybridPipeline
from src.logger import get_logger

_START_TIME = time.monotonic()
_logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Config + singletons (lazy-initialised so tests can monkeypatch before import)
# ---------------------------------------------------------------------------

_config: Optional[dict[str, Any]] = None
_pipeline: Optional[HybridPipeline] = None
_feedback_store: Optional[FeedbackStore] = None

# Rate-limit counters (in-memory; process-scoped)
_total_requests: int = 0
_rate_limited_count: int = 0
_window_start: float = time.monotonic()
_window_requests: int = 0


def _get_config() -> dict[str, Any]:
    """Return the lazily-loaded project config singleton."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def _get_pipeline() -> HybridPipeline:
    """Return the lazily-initialised HybridPipeline singleton."""
    global _pipeline
    if _pipeline is None:
        _pipeline = HybridPipeline(_get_config())
    return _pipeline


def _get_feedback_store() -> FeedbackStore:
    """Return the lazily-initialised FeedbackStore singleton."""
    global _feedback_store
    if _feedback_store is None:
        cfg = _get_config()
        db_path: str = cfg["feedback"]["db_path"]
        min_retrain: int = int(cfg["feedback"]["min_corrections_for_retrain"])
        _feedback_store = FeedbackStore(db_path, min_retrain)
    return _feedback_store


# ---------------------------------------------------------------------------
# App + rate limiter
# ---------------------------------------------------------------------------

cfg_for_init = _get_config()
_rate_limit_str: str = str(cfg_for_init["api"].get("rate_limit_classify", "120/minute"))
_max_payload_mb: int = int(cfg_for_init["api"].get("max_payload_mb", 1))
_cors_origins: list[str] = list(cfg_for_init["api"].get("cors_origins", ["*"]))

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="P1 Hybrid Jailbreak Detector",
    version="0.1.0",
    description="Hybrid LLM jailbreak and prompt injection detector.",
)
app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler,  # type: ignore[arg-type]
)

# Prometheus instrumentation
Instrumentator().instrument(app).expose(app)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])


@app.middleware("http")
async def content_length_guard(request: Request, call_next: Any) -> Response:
    """Reject requests whose Content-Length exceeds the configured MB limit."""
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > _max_payload_mb * 1024 * 1024:
        return JSONResponse(
            status_code=413,
            content={"detail": f"Payload exceeds {_max_payload_mb}MB limit"},
        )
    return await call_next(request)  # type: ignore[return-value,no-any-return]


@app.middleware("http")
async def request_logger(request: Request, call_next: Any) -> Response:
    """Assign a request ID, increment counters, and log every inbound HTTP request."""
    global _total_requests, _window_requests, _window_start
    _total_requests += 1
    _window_requests += 1
    if time.monotonic() - _window_start >= 60:
        _window_start = time.monotonic()
        _window_requests = 1

    request_id = str(uuid.uuid4())
    _logger.info(
        "http_request",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
        },
    )
    response: Response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


@app.exception_handler(ProjectBaseError)
async def project_error_handler(
    request: Request, exc: ProjectBaseError
) -> JSONResponse:
    """Return a 500 JSON response for any unhandled ProjectBaseError."""
    _logger.error("project_error", extra={"error": str(exc)})
    return JSONResponse(status_code=500, content={"detail": str(exc)})


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/v1/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return service health: uptime, memory, and pipeline/model load status."""
    proc = psutil.Process(os.getpid())
    mem_mb = proc.memory_info().rss / (1024 * 1024)
    pipeline_ready = _pipeline is not None
    model_loaded = pipeline_ready and (
        _pipeline._stage_a._model is not None  # type: ignore[union-attr]
    )
    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        pipeline_ready=pipeline_ready,
        uptime_seconds=round(time.monotonic() - _START_TIME, 2),
        memory_mb=round(mem_mb, 2),
    )


@app.post("/api/v1/classify", response_model=ClassifyResponse)
@limiter.limit(_rate_limit_str)
async def classify(request: Request, body: ClassifyRequest) -> ClassifyResponse:
    """Classify a single prompt through the full hybrid pipeline."""
    pipeline = _get_pipeline()
    result = pipeline.classify(body, explain=True)
    _logger.info(
        "classify_result",
        extra={
            "label": result.label,
            "decision": result.decision,
            "confidence": result.confidence,
            "stage_used": result.stage_used,
            "perplexity_score": result.perplexity_score,
        },
    )
    return result


@app.post("/api/v1/classify_batch", response_model=BatchClassifyResponse)
@limiter.limit(_rate_limit_str)
async def classify_batch(
    request: Request, body: BatchClassifyRequest
) -> BatchClassifyResponse:
    """Classify a batch of prompts; every item goes through the full gate stack."""
    pipeline = _get_pipeline()
    responses = pipeline.classify_batch(body.requests)
    return BatchClassifyResponse(responses=responses)


@app.get("/api/v1/classify/stream")
@limiter.limit(_rate_limit_str)
async def classify_stream(
    request: Request,
    user_prompt: str = Query(..., description="Text to classify"),
    external_context: Optional[str] = Query(None),
) -> EventSourceResponse:
    """Emit one SSE event per pipeline stage as classification progresses."""
    pipeline = _get_pipeline()

    async def _event_generator() -> AsyncGenerator[dict[str, Any], None]:
        # Stage 1: Normalize
        normalized, norm_tags = pipeline._normalizer.normalize(user_prompt)
        yield {
            "event": "normalization",
            "data": json.dumps({"applied": list(norm_tags)}),
        }

        # Stage 2: Perplexity
        ppl_result = pipeline._perplexity_fn(normalized, pipeline._config)
        yield {
            "event": "perplexity",
            "data": json.dumps(
                {
                    "score": round(ppl_result.get("perplexity", 0.0), 3),
                    "blocked": ppl_result.get("blocked", False),
                }
            ),
        }
        if ppl_result.get("blocked"):
            yield {
                "event": "decision",
                "data": json.dumps(
                    {"decision": "block", "reason_tags": ["perplexity_anomaly"]}
                ),
            }
            return

        # Stage 3: Similarity
        sim_result = pipeline._similarity.check(normalized)
        yield {
            "event": "similarity",
            "data": json.dumps(
                {
                    "score": round(float(sim_result.get("similarity_score") or 0.0), 3),
                    "blocked": sim_result.get("blocked", False),
                }
            ),
        }
        if sim_result.get("blocked"):
            yield {
                "event": "decision",
                "data": json.dumps(
                    {"decision": "block", "reason_tags": ["known_attack_similarity"]}
                ),
            }
            return

        # Stage 4: Stage A
        classify_req = ClassifyRequest(
            user_prompt=user_prompt,
            external_context=external_context,
        )
        if classify_req.conversation_history:
            full_text = "\n".join(
                list(classify_req.conversation_history) + [normalized]
            )
        else:
            full_text = normalized

        stage_a_result = pipeline._stage_a.classify(full_text)
        yield {
            "event": "stage_a",
            "data": json.dumps(
                {
                    "label": str(stage_a_result.get("label")),
                    "confidence": round(
                        float(stage_a_result.get("confidence", 0.0)), 4
                    ),
                }
            ),
        }

        # Stage 5: Stage B (if escalation)
        stage_b_result = None
        if pipeline._policy.should_escalate(stage_a_result, classify_req, []):
            stage_b_result = pipeline._stage_b.judge(full_text, stage_a_result)
            yield {
                "event": "stage_b",
                "data": json.dumps(
                    {
                        "skipped": False,
                        "is_safe": stage_b_result.get("is_safe", True),
                        "violation_categories": stage_b_result.get(
                            "violation_categories", []
                        ),
                    }
                ),
            }
        else:
            yield {"event": "stage_b", "data": json.dumps({"skipped": True})}

        # Final decision
        final = pipeline._policy.decide(
            stage_a_result=stage_a_result,
            stage_b_result=stage_b_result,
            perplexity_result=ppl_result,
            similarity_result=sim_result,
            request=classify_req,
            reason_tags_in=[],
        )
        yield {
            "event": "decision",
            "data": json.dumps(
                {
                    "decision": final.decision,
                    "reason_tags": final.reason_tags,
                }
            ),
        }

    return EventSourceResponse(_event_generator())


@app.post("/api/v1/feedback")
async def submit_feedback(body: FeedbackRequest) -> dict[str, Any]:
    """Store a human correction for a previous classification decision."""
    store = _get_feedback_store()
    return store.submit_correction(body)


@app.get("/api/v1/feedback/stats")
async def feedback_stats() -> dict[str, Any]:
    """Return aggregated feedback statistics and retrain-readiness flag."""
    store = _get_feedback_store()
    return store.get_stats()


@app.get("/api/v1/rate-limit/stats")
async def rate_limit_stats() -> dict[str, Any]:
    """Return current-window request counts and requests-per-minute estimate."""
    elapsed = time.monotonic() - _window_start
    rpm = (_window_requests / elapsed * 60) if elapsed > 0 else 0.0
    return {
        "total_requests": _total_requests,
        "rate_limited_count": _rate_limited_count,
        "requests_per_minute": round(rpm, 2),
        "current_window": {
            "window_requests": _window_requests,
            "elapsed_seconds": round(elapsed, 2),
        },
    }
