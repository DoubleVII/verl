"""Async adapter for the standalone translation-search-service (no DeepSeek key here)."""

import asyncio
import json
import os
from uuid import uuid4

import httpx

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import (
    OpenAIFunctionSchema,
    OpenAIFunctionToolSchema,
    ToolResponse,
)
from verl.tools.search_post_edit_protocol import (
    TOOL_SCHEMA,
    compact_evidence,
    validate_queries,
)


class SearchFunctionSchema(OpenAIFunctionSchema):
    # The generic tool schema only retains scalar properties, dropping array items.
    # Keep the complete nested JSON Schema locally without changing other tools.
    parameters: dict


class SearchToolSchema(OpenAIFunctionToolSchema):
    function: SearchFunctionSchema


class SearchServiceError(RuntimeError):
    pass


class TranslationSearchTool(BaseTool):
    def __init__(self, config, tool_schema=None):
        if tool_schema is not None:
            raise ValueError("TranslationSearchTool uses its built-in nested schema; omit tool_schema")
        super().__init__(config, tool_schema)
        self.url = config.get("service_url") or os.environ.get("SEARCH_SERVICE_URL", "http://127.0.0.1:8080")
        self.timeout = float(config.get("timeout", 130))
        self.max_response_chars = int(config.get("max_response_chars", 24000))
        concurrency = int(config.get("max_concurrency", 16))
        if self.timeout <= 0 or self.max_response_chars < 1024 or concurrency < 1:
            raise ValueError("Invalid search tool limits")
        self.concurrency = concurrency
        self.semaphore = asyncio.Semaphore(concurrency)
        self.instances = {}
        self.client = None

    def get_openai_tool_schema(self):
        return SearchToolSchema.model_validate(TOOL_SCHEMA)

    async def create(self, instance_id=None, **kwargs):
        options = kwargs.get("create_kwargs", {})
        span_count = options.get("span_count")
        if type(span_count) is not int or span_count < 0:
            raise ValueError("create_kwargs.span_count must be nonnegative")
        instance_id = instance_id or uuid4().hex
        self.instances[instance_id] = span_count
        return instance_id, ToolResponse()

    async def execute(self, instance_id, parameters, **kwargs):
        body = validate_queries(parameters, self.instances[instance_id])
        if self.client is None:
            key = os.environ.get(self.config.get("api_key_env", "SEARCH_SERVICE_API_KEY"), "")
            self.client = httpx.AsyncClient(
                base_url=self.url.rstrip("/"),
                timeout=self.timeout,
                headers={"Authorization": "Bearer " + key} if key else {},
                trust_env=self.config.get("trust_env", True),
                limits=httpx.Limits(
                    max_connections=self.concurrency, max_keepalive_connections=self.concurrency
                ),
            )

        async def request():
            async with self.semaphore:
                response = await self.client.post("/v1/search/batch", json=body)
                response.raise_for_status()
                evidence, metadata = compact_evidence(
                    response.json(),
                    [q["span_id"] for q in body["queries"]],
                    self.max_response_chars,
                )
            text = json.dumps(evidence, ensure_ascii=False, allow_nan=False)
            return text, metadata

        try:
            # Include semaphore queueing in the deadline; compatible with Python 3.10.
            text, metadata = await asyncio.wait_for(request(), timeout=self.timeout)
        except (TimeoutError, httpx.HTTPError, ValueError) as exc:
            raise SearchServiceError(f"search_service_failed:{type(exc).__name__}") from exc
        return (
            ToolResponse(text=text),
            0.0,
            {
                "queries": body["queries"],
                "search_metadata": metadata,
            },
        )

    async def release(self, instance_id, **kwargs):
        self.instances.pop(instance_id, None)

    async def aclose(self):
        if self.client is not None:
            await self.client.aclose()
            self.client = None
