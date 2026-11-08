"""Role → LLMClient router (multi-model)."""

from __future__ import annotations

from cozmo.domain.cost import UsageLedger
from cozmo.domain.ports import LLMClient
from cozmo.domain.roles import ModelRole
from cozmo.infra.llm.factory import build_llm
from cozmo.infra.llm.metered import MeteredLLM
from cozmo.infra.telemetry.tracer import Tracer
from cozmo.settings import Settings


class ModelRouter:
    def __init__(
        self, clients: dict[ModelRole, LLMClient], *, ledger: UsageLedger | None = None
    ) -> None:
        if ModelRole.ORCHESTRATOR not in clients and ModelRole.WORKER not in clients:
            raise ValueError("ModelRouter needs at least orchestrator or worker")
        self._clients = dict(clients)
        self.ledger = ledger

    def client_for(self, role: ModelRole) -> LLMClient:
        if role in self._clients:
            return self._clients[role]
        if ModelRole.WORKER in self._clients:
            return self._clients[ModelRole.WORKER]
        return self._clients[ModelRole.ORCHESTRATOR]

    @property
    def worker(self) -> LLMClient:
        return self.client_for(ModelRole.WORKER)

    @property
    def orchestrator(self) -> LLMClient:
        return self.client_for(ModelRole.ORCHESTRATOR)

    @classmethod
    def from_single(cls, llm: LLMClient) -> ModelRouter:
        return cls({ModelRole.ORCHESTRATOR: llm, ModelRole.WORKER: llm})

    def metered(self, ledger: UsageLedger, model: str, tracer: Tracer) -> ModelRouter:
        """Same clients, each one measured against *ledger*."""
        wrapped: dict[ModelRole, LLMClient] = {
            role: MeteredLLM(client, model=model, role=role.value, ledger=ledger, tracer=tracer)
            for role, client in self._clients.items()
        }
        return ModelRouter(wrapped, ledger=ledger)


def build_llm_bundle(
    settings: Settings,
    *,
    ledger: UsageLedger | None = None,
    tracer: Tracer | None = None,
) -> ModelRouter:
    """Orchestrator = settings.model; worker = worker_model or same. All calls are metered."""
    ledger = ledger or UsageLedger(settings.model_prices)
    tracer = tracer or Tracer(None, enabled=False)
    worker_model = (settings.worker_model or "").strip() or settings.model

    def client(model: str, role: ModelRole) -> LLMClient:
        inner = build_llm(
            settings.model_copy(update={"model": model}), on_retry=ledger.record_retry
        )
        return MeteredLLM(inner, model=model, role=role.value, ledger=ledger, tracer=tracer)

    return ModelRouter(
        {
            ModelRole.ORCHESTRATOR: client(settings.model, ModelRole.ORCHESTRATOR),
            ModelRole.WORKER: client(worker_model, ModelRole.WORKER),
        },
        ledger=ledger,
    )
