from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from berkeley_marl.agents import DQNAgent, RandomPolicy
from berkeley_marl.envs import BerkeleyForestEnv, EnvConfig
from berkeley_marl.utils.seeding import seed_everything


def run_eval(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    env = BerkeleyForestEnv(
        EnvConfig(
            condition=args.condition,
            grid_size=args.grid_size,
            n_agents=args.n_agents,
            max_steps=args.max_steps,
            observation_radius=args.observation_radius,
            seed=args.seed,
        )
    )

    if args.model:
        policy = DQNAgent.load(args.model, device=args.device)
        act = lambda obs: policy.act(obs, epsilon=0.0)
    else:
        random_policy = RandomPolicy(env.n_actions, seed=args.seed)
        act = random_policy.act

    returns = []
    successes = []
    lengths = []

    for episode in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode)
        episode_return = 0.0
        success = False
        for step in range(args.max_steps):
            actions = {name: act(agent_obs) for name, agent_obs in obs.items()}
            obs, rewards, terminations, truncations, infos = env.step(actions)
            episode_return += next(iter(rewards.values()))
            success = success or any(info["success"] for info in infos.values())
            if any(terminations.values()) or any(truncations.values()):
                break
        returns.append(episode_return)
        successes.append(float(success))
        lengths.append(step + 1)

    print(f"condition: {args.condition}")
    print(f"episodes: {args.episodes}")
    print(f"mean_return: {sum(returns) / len(returns):.3f}")
    print(f"success_rate: {sum(successes) / len(successes):.3f}")
    print(f"mean_length: {sum(lengths) / len(lengths):.3f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Berkeley Forest policy.")
    parser.add_argument("--condition", choices=["berkeley", "materialist"], required=True)
    parser.add_argument("--model", default=None, help="Optional path to a .pt checkpoint. Uses random policy when omitted.")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--grid-size", type=int, default=7)
    parser.add_argument("--n-agents", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--observation-radius", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    return parser


if __name__ == "__main__":
    run_eval(build_parser().parse_args())
