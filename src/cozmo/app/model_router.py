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
        # Fallbacks: verifier → orchestrator → worker
        if role == ModelRole.VERIFIER and ModelRole.ORCHESTRATOR in self._clients:
            return self._clients[ModelRole.ORCHESTRATOR]
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
    """
    Orchestrator = settings.model; worker = worker_model or same.
    Verifier optional — omitted unless verifier_model is set.
    """
    orchestrator = build_llm(settings)
    worker_model = (settings.worker_model or "").strip() or None
    if worker_model and worker_model != settings.model:
        worker = build_llm(settings.model_copy(update={"model": worker_model}))
    else:
        worker = orchestrator

    clients: dict[ModelRole, LLMClient] = {
        ModelRole.ORCHESTRATOR: orchestrator,
        ModelRole.WORKER: worker,
    }
    verifier_model = (settings.verifier_model or "").strip() or None
    if verifier_model:
        if verifier_model == settings.model:
            clients[ModelRole.VERIFIER] = orchestrator
        elif verifier_model == worker_model:
            clients[ModelRole.VERIFIER] = worker
        else:
            clients[ModelRole.VERIFIER] = build_llm(
                settings.model_copy(update={"model": verifier_model})
            )
    return ModelRouter(clients)
