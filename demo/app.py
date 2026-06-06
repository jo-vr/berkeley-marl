from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from berkeley_marl.agents import RandomPolicy
from berkeley_marl.envs import BerkeleyForestEnv, EnvConfig

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Berkeley MARL Demo")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_envs: Dict[str, BerkeleyForestEnv] = {}
_policies: Dict[str, RandomPolicy] = {}


class ResetRequest(BaseModel):
    condition: str = "berkeley"
    seed: int | None = None


class StepRequest(BaseModel):
    condition: str = "berkeley"
    actions: Dict[str, int] | None = None


def get_env(condition: str) -> BerkeleyForestEnv:
    if condition not in {"berkeley", "materialist"}:
        raise ValueError("condition must be 'berkeley' or 'materialist'")
    if condition not in _envs:
        _envs[condition] = BerkeleyForestEnv(EnvConfig(condition=condition, seed=123))
        _envs[condition].reset(seed=123)
        _policies[condition] = RandomPolicy(_envs[condition].n_actions, seed=123)
    return _envs[condition]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/reset")
def reset(request: ResetRequest) -> dict:
    env = BerkeleyForestEnv(EnvConfig(condition=request.condition, seed=request.seed))
    env.reset(seed=request.seed)
    _envs[request.condition] = env
    _policies[request.condition] = RandomPolicy(env.n_actions, seed=request.seed)
    return {"state": env.state_for_demo(), "message": "reset"}


@app.post("/api/step")
def step(request: StepRequest) -> dict:
    env = get_env(request.condition)
    if env.done:
        env.reset()
    obs = env._observations()  # demo-only convenience
    policy = _policies[request.condition]
    actions = request.actions or {agent: policy.act(agent_obs) for agent, agent_obs in obs.items()}
    _, rewards, terminations, truncations, infos = env.step(actions)
    return {
        "state": env.state_for_demo(),
        "actions": actions,
        "reward": next(iter(rewards.values())),
        "done": any(terminations.values()) or any(truncations.values()),
        "success": any(info["success"] for info in infos.values()),
    }
