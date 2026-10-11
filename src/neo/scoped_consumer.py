"""Original-parent CAR inference binding supplied by a trusted launcher."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any


@dataclass(frozen=True, init=False)
class ScopedConsumer:
    """Keep an admitted facade and its private storage identity together.

    Construct through ``from_issued``; a context label alone grants nothing.
    The parent keeps its coordination socket. This child never claims an inbox.
    """
    parent_agent_id: str
    resource_key: str
    context_json: str
    private_root: Path
    runtime: Any = field(repr=False, compare=False)

    @classmethod
    def from_issued(cls, daemon_url: str, receipt_json: str,
                    parent_agent_id: str, resource_key: str, work_context: dict,
                    private_root: Path, *, factory=None) -> "ScopedConsumer":
        context_json = json.dumps(work_context, sort_keys=True, separators=(",", ":"))
        if factory is None:
            from car_runtime import CarRuntime
            factory = CarRuntime.with_scoped_consumer
        # The SDK validates exact parent/resource/context, role, expiry and the
        # daemon receipt before returning this restricted inference facade.
        runtime = factory(daemon_url, receipt_json, parent_agent_id,
                          resource_key, context_json)
        binding = object.__new__(cls)
        for name, value in (("parent_agent_id", parent_agent_id),
                            ("resource_key", resource_key),
                            ("context_json", context_json),
                            ("private_root", Path(private_root).resolve()),
                            ("runtime", runtime)):
            object.__setattr__(binding, name, value)
        return binding

    @property
    def state_root(self) -> Path:
        identity = json.dumps([self.parent_agent_id, self.resource_key,
                               self.context_json], separators=(",", ":"))
        return self.private_root / hashlib.sha256(identity.encode()).hexdigest()

    def authorize(self) -> None:
        # Routing is decision-only. Its actual granted RPC revalidates this
        # child before any local private read; refusal propagates unchanged.
        # A Neo grant must include routing as well as inference.
        self.runtime.route_model("Validate original Neo consumer authority",
                                 '{"task":"code"}')

    def prepare_state_root(self) -> Path:
        self.authorize()
        root = self.state_root
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = root.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError("Scoped Neo state must be an owned private directory")
        return root
