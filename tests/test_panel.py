"""Tests for panel model-diversity planning (route_model + exclude_models)."""

import json

from neo.panel import (
    build_role_factory,
    capable_model_count,
    plan_role_models,
)


def make_route_fn(pool):
    """Fake ``route_model``: rank pool by score, honor exclude_models, mark the
    top one selected — mirroring CAR's decision JSON."""

    def route(prompt, intent_json):
        intent = json.loads(intent_json)
        exclude = set(intent.get("exclude_models", []))
        ranked = sorted(
            (m for m in pool if m["id"] not in exclude),
            key=lambda m: m["score"],
            reverse=True,
        )
        cands = [
            {"model_id": m["id"], "score": m["score"], "reliability": m["score"],
             "selected": i == 0, "in_band": True}
            for i, m in enumerate(ranked)
        ]
        return json.dumps({"selected": cands[0]["model_id"] if cands else None,
                           "candidates": cands})

    return route


POOL = [
    {"id": "gpt", "score": 0.90},
    {"id": "claude", "score": 0.85},
    {"id": "qwen-local", "score": 0.60},
]


def test_capable_model_count_reads_candidates():
    assert capable_model_count(make_route_fn(POOL)) == 3
    assert capable_model_count(make_route_fn([])) == 0


def test_critic_is_forced_off_the_coder_model():
    plan = plan_role_models(make_route_fn(POOL), "task")
    # coder gets the top model; critic must be excluded off it -> different model
    assert plan["coder"] == "gpt"
    assert plan["critic"] != plan["coder"]
    assert plan["critic"] == "claude"      # next best after excluding gpt


def test_single_model_pool_cannot_diversify_critic():
    # only one capable model -> exclude leaves nothing -> critic unassigned
    plan = plan_role_models(make_route_fn([{"id": "solo", "score": 0.9}]), "task")
    assert plan["coder"] == "solo"
    assert "critic" not in plan            # no distinct model available
    assert capable_model_count(make_route_fn([{"id": "solo", "score": 0.9}])) == 1


def test_route_failures_yield_empty_plan_not_crash():
    def broken(prompt, intent_json):
        return "not json"

    assert plan_role_models(broken, "t") == {}
    assert capable_model_count(broken) == 0


def test_factory_pins_distinct_adapters_with_fallback():
    built = []

    def adapter_for_model(m):
        built.append(m)
        return f"adapter:{m}"

    plan = {"coder": "gpt", "critic": "claude"}  # planner/judge unassigned
    factory = build_role_factory(plan, adapter_for_model, fallback_adapter="FALLBACK")
    assert factory("coder") == "adapter:gpt"
    assert factory("critic") == "adapter:claude"
    assert factory("planner") == "FALLBACK"      # unassigned -> fallback
    # adapters are cached (built once each)
    factory("coder")
    assert built == ["gpt", "claude"]
