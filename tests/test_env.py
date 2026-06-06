from berkeley_marl.envs import BerkeleyForestEnv, EnvConfig


def test_reset_returns_observations_for_all_agents():
    env = BerkeleyForestEnv(EnvConfig(condition="berkeley", n_agents=2, seed=1))
    obs, info = env.reset(seed=1)
    assert set(obs) == {"agent_0", "agent_1"}
    assert info["condition"] == "berkeley"
    assert all(value.shape == (env.observation_size,) for value in obs.values())


def test_materialist_success_on_tree_reach():
    env = BerkeleyForestEnv(EnvConfig(condition="materialist", n_agents=1, seed=2))
    env.reset(seed=2)
    env.agent_positions["agent_0"] = env.tree_position
    _, _, terminations, _, infos = env.step({"agent_0": 0})
    assert terminations["agent_0"] is True
    assert infos["agent_0"]["success"] is True


def test_berkeley_requires_report_while_perceiving():
    env = BerkeleyForestEnv(EnvConfig(condition="berkeley", n_agents=1, seed=3, observation_radius=1))
    env.reset(seed=3)
    env.agent_positions["agent_0"] = env.tree_position
    _, _, terminations, _, infos = env.step({"agent_0": env.REPORT_ACTION})
    assert terminations["agent_0"] is True
    assert infos["agent_0"]["success"] is True
