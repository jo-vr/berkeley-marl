from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from berkeley_marl.agents import DQNAgent, DQNConfig
from berkeley_marl.envs import BerkeleyForestEnv, EnvConfig
from berkeley_marl.utils.logging import append_csv
from berkeley_marl.utils.seeding import seed_everything


def linear_epsilon(episode: int, episodes: int, start: float, end: float) -> float:
    if episodes <= 1:
        return end
    fraction = min(max(episode / float(episodes - 1), 0.0), 1.0)
    return start + fraction * (end - start)


def run_training(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

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
    agent = DQNAgent(
        DQNConfig(
            observation_size=env.observation_size,
            n_actions=env.n_actions,
            hidden_size=args.hidden_size,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            batch_size=args.batch_size,
            device=args.device,
        )
    )

    log_path = results_dir / f"{args.condition}_seed{args.seed}.csv"
    fieldnames = ["episode", "condition", "seed", "return", "success", "steps", "epsilon", "loss"]

    for episode in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode)
        epsilon = linear_epsilon(episode, args.episodes, args.epsilon_start, args.epsilon_end)
        episode_return = 0.0
        final_success = False
        losses: list[float] = []

        for step in range(args.max_steps):
            actions = {name: agent.act(agent_obs, epsilon=epsilon) for name, agent_obs in obs.items()}
            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            done = any(terminations.values()) or any(truncations.values())
            shared_reward = next(iter(rewards.values()))
            episode_return += shared_reward
            final_success = any(info["success"] for info in infos.values())

            for name in obs.keys():
                agent.remember((obs[name], actions[name], shared_reward, next_obs[name], done))

            loss = agent.update()
            if loss is not None:
                losses.append(loss)

            obs = next_obs
            if done:
                break

        append_csv(
            log_path,
            {
                "episode": episode,
                "condition": args.condition,
                "seed": args.seed,
                "return": round(episode_return, 6),
                "success": int(final_success),
                "steps": step + 1,
                "epsilon": round(epsilon, 6),
                "loss": round(sum(losses) / len(losses), 6) if losses else "",
            },
            fieldnames,
        )

        if (episode + 1) % args.log_interval == 0:
            print(
                f"[{args.condition} seed={args.seed}] episode={episode + 1}/{args.episodes} "
                f"return={episode_return:.2f} success={int(final_success)} epsilon={epsilon:.3f}"
            )

    checkpoint_path = results_dir / f"{args.condition}_seed{args.seed}.pt"
    agent.save(
        str(checkpoint_path),
        metadata={
            "condition": args.condition,
            "seed": args.seed,
            "episodes": args.episodes,
            "grid_size": args.grid_size,
            "n_agents": args.n_agents,
        },
    )
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved log: {log_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a shared DQN baseline on Berkeley Forest.")
    parser.add_argument("--condition", choices=["berkeley", "materialist"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--grid-size", type=int, default=7)
    parser.add_argument("--n-agents", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--observation-radius", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--log-interval", type=int, default=25)
    return parser


if __name__ == "__main__":
    run_training(build_parser().parse_args())
