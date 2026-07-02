from typing import AsyncGenerator


async def classify_stream(  # type: ignore[type-arg]
    request: dict,
) -> AsyncGenerator[str, None]:
    """SSE streaming endpoint for stage-by-stage results."""
    yield ""
