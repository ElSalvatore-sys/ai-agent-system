import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.core.config import settings
from app.core.logger import LoggerMixin


class BudgetManagerMiddleware(BaseHTTPMiddleware, LoggerMixin):
    """Middleware that enforces per-user monthly budget limits.

    The middleware expects downstream handlers (e.g. AdvancedAIOrchestrator or
    CostTrackingMiddleware) to attach the actual API cost of the request in
    the response header ``X-API-Cost`` **in USD**.  If that header is missing
    the request is considered free.

    A lightweight in-memory dictionary is used for the running tally.  This can
    be replaced with Redis for horizontal scaling – just swap the
    ``self._user_spend`` accessors with async Redis ``INCRBYFLOAT`` calls.
    """

    HEALTH_ENDPOINTS = {"/", "/health", "/docs", "/redoc", "/openapi.json"}

    def __init__(self, app):
        super().__init__(app)
        # user_id -> accumulated spend for current UTC month (float USD)
        self._user_spend = defaultdict(float)
        # user_id -> month stamp (YYYY-MM) used to detect month rollover
        self._user_month = defaultdict(lambda: self._current_month())

    # ------------------------------------------------------------------
    # Middleware entrypoint
    # ------------------------------------------------------------------
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip health/static
        if request.url.path in self.HEALTH_ENDPOINTS:
            return await call_next(request)

        user_id = getattr(request.state, "user_id", None)
        # If unauthenticated or no budget enforcement required – proceed.
        if user_id is None:
            return await call_next(request)

        # Ensure month context is up-to-date.
        self._rollover_month(user_id)

        limit = settings.MONTHLY_COST_LIMIT
        spent = self._user_spend[user_id]
        if spent >= limit:
            # User has exhausted budget – reject request.
            return JSONResponse(
                status_code=402,
                content={
                    "error": {
                        "code": 402,
                        "type": "budget_exceeded",
                        "message": f"Monthly budget limit ${limit:.2f} reached"
                    }
                }
            )

        # Process request
        response: Response = await call_next(request)

        # Extract actual cost from response header (if provided)
        cost = 0.0
        if hasattr(response, "headers") and "X-API-Cost" in response.headers:
            try:
                cost = float(response.headers["X-API-Cost"])
            except (TypeError, ValueError):
                cost = 0.0

        # Update spend
        self._user_spend[user_id] += cost
        remaining = max(limit - self._user_spend[user_id], 0.0)

        # Add budget headers for client awareness
        response.headers["X-Budget-Limit"] = f"{limit:.2f}"
        response.headers["X-Budget-Remaining"] = f"{remaining:.4f}"
        if remaining / limit <= 0.1:
            response.headers["X-Budget-Warning"] = "true"

        # Debug log
        self.logger.debug(
            "budget_update",
            extra={
                "user_id": user_id,
                "cost": cost,
                "spent": self._user_spend[user_id],
                "remaining": remaining,
            },
        )
        return response

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _current_month() -> str:
        now = datetime.now(timezone.utc)
        return now.strftime("%Y-%m")

    def _rollover_month(self, user_id: int) -> None:
        """Reset spend counter if we are in a new month."""
        month_now = self._current_month()
        if self._user_month[user_id] != month_now:
            self._user_spend[user_id] = 0.0
            self._user_month[user_id] = month_now

    # ------------------------------------------------------------------
    # Public inspection API (useful for admin endpoints/tests)
    # ------------------------------------------------------------------
    def get_user_spend(self, user_id: int) -> float:
        return self._user_spend.get(user_id, 0.0)

    def get_budget_stats(self):
        return {
            "total_users": len(self._user_spend),
            "total_spend": sum(self._user_spend.values()),
        }
