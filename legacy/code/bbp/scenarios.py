"""Routing policies shared across detectors.

Each route(pid, rng) callable selects the serving source for one request."""
import numpy as np


def make_policies(P, rhos=(0.05, 0.1, 0.25, 0.5, 1.0)):
    """P: {pid: prompt dict}. Returns {name: route_fn}."""
    pol = {}
    for r in rhos:
        pol[f"random_{r}"] = (lambda r: (lambda pid, rng: rng.random() < r))(r)
    # cost-saving routers: send 'cheap-looking' traffic to the substitute, keep the rest genuine
    pol["route_chat_only"] = lambda pid, rng: P[pid]["domain"] == "wildchat"
    pol["route_zh_only"] = lambda pid, rng: P[pid]["lang"] == "zh"
    lens = np.array([len(p["text"]) for p in P.values()])
    med = float(np.median(lens))
    pol["route_short_prompts"] = lambda pid, rng: len(P[pid]["text"]) < med
    pol["route_code_math_only"] = lambda pid, rng: P[pid]["domain"] in ("bigcodebench", "math")
    return pol
