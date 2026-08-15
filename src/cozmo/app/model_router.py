"""Role → LLMClient router (multi-model)."""

from __future__ import annotations

from cozmo.domain.ports import LLMClient
from cozmo.domain.roles import ModelRole
from cozmo.infra.llm.factory import build_llm
from cozmo.settings import Settings


class ModelRouter:
    def __init__(self, clients: dict[ModelRole, LLMClient]) -> None:
        if ModelRole.ORCHESTRATOR not in clients and ModelRole.WORKER not in clients:
            raise ValueError("ModelRouter needs at least orchestrator or worker")
        self._clients = dict(clients)

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


def build_llm_bundle(settings: Settings) -> ModelRouter:
    """Orchestrator = settings.model; worker = worker_model or same."""
    orchestrator = build_llm(settings)
    worker_model = (settings.worker_model or "").strip() or None
    if worker_model and worker_model != settings.model:
        worker = build_llm(settings.model_copy(update={"model": worker_model}))
    else:
        worker = orchestrator

    return ModelRouter(
        {
            ModelRole.ORCHESTRATOR: orchestrator,
            ModelRole.WORKER: worker,
        }
    )
