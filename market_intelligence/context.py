from contextvars import ContextVar

request_id_context: ContextVar[str] = ContextVar("market_intelligence_request_id", default="background")
