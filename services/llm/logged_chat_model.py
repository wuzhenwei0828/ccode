import logging
from collections.abc import AsyncIterator, Iterator
from typing import Any

logger = logging.getLogger(__name__)


class LoggedChatModel:
    """Stable logging proxy around a chat model instance."""

    def __init__(self, inner: Any, provider: str, model_name: str):
        self.inner = inner
        self.provider = provider
        self.model_name = model_name

    def invoke(self, payload: Any, **kwargs: Any) -> Any:
        logger.info(
            "llm invoke request provider=%s model=%s payload=%s",
            self.provider,
            self.model_name,
            payload,
        )
        if kwargs:
            logger.info(
                "llm invoke kwargs provider=%s model=%s kwargs=%s",
                self.provider,
                self.model_name,
                kwargs,
            )
        response = self.inner.invoke(payload, **kwargs)
        logger.info(
            "llm invoke response provider=%s model=%s raw_response=%s",
            self.provider,
            self.model_name,
            response,
        )
        return response

    def stream(self, payload: Any, **kwargs: Any) -> Iterator[Any]:
        logger.info(
            "llm stream request provider=%s model=%s payload=%s",
            self.provider,
            self.model_name,
            payload,
        )
        if kwargs:
            logger.info(
                "llm stream kwargs provider=%s model=%s kwargs=%s",
                self.provider,
                self.model_name,
                kwargs,
            )
        for chunk in self.inner.stream(payload, **kwargs):
            logger.info(
                "llm stream chunk provider=%s model=%s raw_chunk=%s",
                self.provider,
                self.model_name,
                chunk,
            )
            yield chunk

    async def astream(self, payload: Any, **kwargs: Any) -> AsyncIterator[Any]:
        logger.info(
            "llm astream request provider=%s model=%s payload=%s",
            self.provider,
            self.model_name,
            payload,
        )
        if kwargs:
            logger.info(
                "llm astream kwargs provider=%s model=%s kwargs=%s",
                self.provider,
                self.model_name,
                kwargs,
            )
        async for chunk in self.inner.astream(payload, **kwargs):
            logger.info(
                "llm astream chunk provider=%s model=%s raw_chunk=%s",
                self.provider,
                self.model_name,
                chunk,
            )
            yield chunk

    def bind_tools(self, tools: list[Any]) -> "LoggedChatModel":
        logger.info(
            "llm bind_tools provider=%s model=%s tools=%s",
            self.provider,
            self.model_name,
            tools,
        )
        bound_inner = self.inner.bind_tools(tools)
        return LoggedChatModel(
            inner=bound_inner,
            provider=self.provider,
            model_name=self.model_name,
        )

    def __getattr__(self, item: str) -> Any:
        return getattr(self.inner, item)
