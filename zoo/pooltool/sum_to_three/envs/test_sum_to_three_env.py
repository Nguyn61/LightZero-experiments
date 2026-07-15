import math
import pytest
import numpy as np
from zoo.pooltool.datatypes import Spaces
from zoo.pooltool.image_representation import PygameRenderer, RenderConfig
from zoo.pooltool.sum_to_three.envs.sum_to_three_env import (
    Bounds,
    SumToThreeEnv,
    StartDistribution,
    SumToThreeSimulator,
    create_initial_state,
    get_action_space,
    sample_initial_positions,
)

import pooltool as pt
from zoo.pooltool.sum_to_three.envs.utils import (
    ObservationType,
    ShotOutcome,
    binary_from_outcome,
    event_aligned_from_outcome,
    get_coordinate_obs_space,
    get_image_obs_space,
    get_reward_space,
)
from easydict import EasyDict

np.random.seed(42)


@pytest.fixture
def env_config():
    return EasyDict(
        {
            "env_name": "PoolTool-SumToThree",
            "episode_length": 10,
            "reward_algorithm": "binary",
            "action_V0_low": 0.3,
            "action_V0_high": 3.0,
            "action_angle_low": -70,
            "action_angle_high": 70,
            "observation_type": "coordinate",
        }
    )


@pytest.fixture
def env(env_config) -> SumToThreeEnv:
    return SumToThreeEnv(env_config)


@pytest.fixture
def st3_coord() -> SumToThreeSimulator:
    state = create_initial_state(random_pos=False)
    observation_space = get_coordinate_obs_space(state.system)
    action_space = get_action_space(
        Bounds(0.3, 3.0),
        Bounds(-70, 70),
    )
    reward_space = get_reward_space(
        "binary",
    )
    spaces = Spaces(
        observation_space,
        action_space,
        reward_space,
    )

    return SumToThreeSimulator(
        state,
        spaces,
        observation_type=ObservationType.COORDINATE,
    )


@pytest.fixture
def st3_image() -> SumToThreeSimulator:
    state = create_initial_state(random_pos=False)
    render_config = RenderConfig.default()
    observation_space = get_image_obs_space(render_config)
    renderer = PygameRenderer.build(state.system.table, render_config)
    renderer.set_state(state)
    renderer.init()

    action_space = get_action_space(
        Bounds(0.3, 3.0),
        Bounds(-70, 70),
    )
    reward_space = get_reward_space(
        "binary",
    )
    spaces = Spaces(
        observation_space,
        action_space,
        reward_space,
    )

    return SumToThreeSimulator(
        state,
        spaces,
        observation_type=ObservationType.IMAGE,
        renderer=renderer,
    )


def _random_normalized_action():
    return np.random.uniform(low=-1.0, high=1.0, size=2).astype(np.float32)


def test_create_initial_state():
    state = create_initial_state(random_pos=False)

    # Total set of balls equals cue and object
    assert {"cue", "object"} == set(state.system.balls.keys())

    # Cue ball is at starting position
    expected = np.array(
        [
            state.system.table.w / 2,
            state.system.table.l / 4,
            state.system.balls["cue"].params.R,
        ],
        dtype=np.float64,
    )
    assert np.allclose(state.system.balls["cue"].xyz, expected, rtol=1e-3)

    # Object ball is at starting position
    expected = np.array(
        [
            state.system.table.w / 2,
            state.system.table.l / 4 * 3,
            state.system.balls["object"].params.R,
        ],
        dtype=np.float64,
    )
    assert np.allclose(state.system.balls["object"].xyz, expected, rtol=1e-3)


def test_create_initial_state_random():
    state = create_initial_state(random_pos=True)
    length = state.system.table.l
    width = state.system.table.w

    assert state.system.balls["cue"].params.R == state.system.balls["object"].params.R
    R = state.system.balls["cue"].params.R

    # Positions are random, so they don't match the starting positions
    expected = np.array([width / 2, length / 4, R], dtype=np.float64)
    assert not np.allclose(state.system.balls["cue"].xyz, expected, rtol=1e-3)

    expected = np.array([width / 2, length / 4 * 3, R], dtype=np.float64)
    assert not np.allclose(state.system.balls["object"].xyz, expected, rtol=1e-3)

    # Assert balls are within the table bounds
    for _ in range(100):
        state = create_initial_state(random_pos=True)
        cue_ball_pos = state.system.balls["cue"].xyz
        object_ball_pos = state.system.balls["object"].xyz

        assert R < cue_ball_pos[0] < width - R
        assert R < cue_ball_pos[1] < length - R
        assert R < object_ball_pos[0] < width - R
        assert R < object_ball_pos[1] < length - R


def test_set_action(st3_coord: SumToThreeSimulator):
    # Define a test action
    test_action = np.array([1.5, -30], dtype=np.float32)  # Example values

    # Set the action
    st3_coord.set_action(action=test_action)

    # Check if the action was set correctly
    assert math.isclose(st3_coord.state.system.cue.V0, test_action[0], rel_tol=1e-3)
    assert math.isclose(
        st3_coord.state.system.cue.phi,
        pt.aim.at_ball(st3_coord.state.system, "object", cut=test_action[1]),
        rel_tol=1e-3,
    )


def test_observation_array_coordinate(st3_coord: SumToThreeSimulator):
    obs_array = st3_coord.observation_array()
    assert obs_array.shape == (4,)
    assert isinstance(obs_array, np.ndarray)

    # Each observation element matches the order of ball coordinates
    assert math.isclose(
        obs_array[0], st3_coord.state.system.balls["cue"].xyz[0], rel_tol=1e-3
    )
    assert math.isclose(
        obs_array[1], st3_coord.state.system.balls["cue"].xyz[1], rel_tol=1e-3
    )
    assert math.isclose(
        obs_array[2], st3_coord.state.system.balls["object"].xyz[0], rel_tol=1e-3
    )
    assert math.isclose(
        obs_array[3], st3_coord.state.system.balls["object"].xyz[1], rel_tol=1e-3
    )


def test_observation_array_image(st3_image: SumToThreeSimulator):
    # Grab an observation
    obs_array = st3_image.observation_array()

    # 3-dimensional output
    assert obs_array.ndim == 3

    assert st3_image.renderer is not None

    # Shape matches what render config specifies
    assert obs_array.shape == st3_image.renderer.render_config.observation_shape
    assert obs_array.shape[0] == st3_image.renderer.render_config.channels
    assert obs_array.shape[1] == st3_image.renderer.render_config.px
    assert obs_array.shape[2] == st3_image.renderer.render_config.px // 2

    # Further, ensure that the rendered image matches the renderer's output
    st3_image.renderer.set_state(st3_image.state)
    rendered_image = st3_image.renderer.observation()

    # Ensure that the rendered image shape matches the observed array shape
    assert obs_array.shape == rendered_image.shape

    # The rendered image and observation array should be almost identical
    assert np.allclose(obs_array, rendered_image, atol=1e-5)


def test_observation_types():
    state = create_initial_state(random_pos=False)
    render_config = RenderConfig.default()
    renderer = PygameRenderer.build(state.system.table, render_config)
    renderer.set_state(state)
    renderer.init()

    for observation_type in ObservationType:
        if observation_type == ObservationType.COORDINATE:
            observation_space = get_coordinate_obs_space(state.system)
        elif observation_type == ObservationType.IMAGE:
            observation_space = get_image_obs_space(render_config)
        else:
            raise NotImplementedError()

        spaces = Spaces(
            observation_space,
            get_action_space(Bounds(0.3, 3.0), Bounds(-70, 70)),
            get_reward_space("binary"),
        )
        simulator = SumToThreeSimulator(
            state,
            spaces,
            observation_type=observation_type,
            renderer=renderer if observation_type == ObservationType.IMAGE else None,
        )

        obs = simulator.observation_array()
        assert obs.shape == observation_space.shape


def test_unnormalized_action_to_step(env: SumToThreeEnv):
    """Make sure `step` fails when unnormalized actions are passed

    For whatever reason, `train_muzero` passes normalized actions [-1, 1] to `step`,
    that are rescaled to the proper action space within the `step` function. This test
    exists so that if the upstream behaviour is changed (AKA the `train_muzero` routine
    passes properly scaled actions to `step`), this test will fail.
    """
    env.reset()

    with pytest.raises(AssertionError):
        env.step(np.array([1.4, -40.0], dtype=np.float32))


def test_step_progresses_state(env: SumToThreeEnv):
    # Initialize the environment
    assert not env._init_flag
    step_0 = env.reset()
    assert env._init_flag

    # Take two steps
    step_1 = env.step(_random_normalized_action())
    step_2 = env.step(_random_normalized_action())

    # Check that each step progresses the state
    assert (
        np.allclose(step_0["observation"], step_1.obs["observation"], atol=1e-3)
        is False
    )
    assert (
        np.allclose(step_1.obs["observation"], step_2.obs["observation"], atol=1e-3)
        is False
    )

    # Check that the episode progresses
    assert not step_1.done
    assert not step_2.done
    assert env._tracked_stats.eval_episode_length == 2


def test_reset(env: SumToThreeEnv):
    # Initialize the env
    assert not env._init_flag
    env.reset()
    assert env._init_flag

    # Make a copy of the system for later use
    system_initial = env._env.state.system.copy()

    # Finish an episode
    for _ in range(env.cfg.episode_length):  # type: ignore
        step = env.step(_random_normalized_action())
    assert step.done  # type: ignore

    # Grab the system, as well as copies
    system = env._env.state.system
    system_copy = env._env.state.system.copy()

    # Now reset the state
    env.reset()

    # Stats are reset
    assert env._tracked_stats.eval_episode_length == 0
    assert env._tracked_stats.eval_episode_return == 0

    # Resetting yields the same system, but with a reset state
    assert env._env.state.system is system
    assert env._env.state.system != system_copy
    assert env._env.state.system == system_initial


def test_reward_from_outcome_lookup():
    cases = [
        (ShotOutcome(False, 0), 0.0, 0.0),
        (ShotOutcome(True, 0), 0.0, 0.1),
        (ShotOutcome(True, 1), 0.0, 0.2),
        (ShotOutcome(True, 2), 0.0, 0.3),
        (ShotOutcome(True, 3), 1.0, 1.0),
        (ShotOutcome(True, 4), 0.0, 0.1),
    ]
    for outcome, expected_binary, expected_event_aligned in cases:
        assert binary_from_outcome(outcome) == expected_binary
        assert event_aligned_from_outcome(outcome) == expected_event_aligned


def test_event_aligned_reward_space():
    reward_space = get_reward_space("event_aligned")
    assert np.all(reward_space.low == 0.0)
    assert np.all(reward_space.high == 1.0)


@pytest.mark.parametrize("distribution", [StartDistribution.LOCAL, StartDistribution.BROAD])
def test_sample_initial_positions_are_valid(distribution):
    state = create_initial_state(random_pos=False)
    system = state.system
    R = system.balls["cue"].params.R
    margin = 0.005
    rng = np.random.default_rng(123)

    for _ in range(100):
        cue_pos, object_pos = sample_initial_positions(
            system,
            distribution,
            rng,
            separation_margin=margin,
        )
        assert R <= cue_pos[0] <= system.table.w - R
        assert R <= cue_pos[1] <= system.table.l - R
        assert R <= object_pos[0] <= system.table.w - R
        assert R <= object_pos[1] <= system.table.l - R
        assert np.linalg.norm(cue_pos[:2] - object_pos[:2]) >= 2 * R + margin


def test_legacy_random_initial_state_honors_numpy_seed():
    np.random.seed(7)
    state_a = create_initial_state(random_pos=True)
    positions_a = [ball.xyz.copy() for ball in state_a.system.balls.values()]

    np.random.seed(7)
    state_b = create_initial_state(random_pos=True)
    positions_b = [ball.xyz.copy() for ball in state_b.system.balls.values()]

    assert all(np.allclose(a, b) for a, b in zip(positions_a, positions_b))


def test_start_sampling_is_seeded():
    state = create_initial_state(random_pos=False)
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)
    rng_c = np.random.default_rng(8)

    positions_a = sample_initial_positions(state.system, StartDistribution.BROAD, rng_a)
    positions_b = sample_initial_positions(state.system, StartDistribution.BROAD, rng_b)
    positions_c = sample_initial_positions(state.system, StartDistribution.BROAD, rng_c)

    assert all(np.allclose(a, b) for a, b in zip(positions_a, positions_b))
    assert any(not np.allclose(a, c) for a, c in zip(positions_a, positions_c))


def test_curriculum_stage_progression(env_config):
    env_config.update(
        start_distribution="curriculum",
        curriculum_enabled=True,
        curriculum_total_episodes=4,
        curriculum_canonical_fraction=0.25,
        curriculum_local_fraction=0.50,
    )
    curriculum_env = SumToThreeEnv(env_config)
    curriculum_env.seed(5, dynamic_seed=False)

    observed = []
    for _ in range(4):
        curriculum_env.reset()
        observed.append(curriculum_env._current_start_distribution)

    assert observed == [
        StartDistribution.CANONICAL,
        StartDistribution.LOCAL,
        StartDistribution.LOCAL,
        StartDistribution.BROAD,
    ]


def test_collector_config_preserves_explicit_curriculum_episode_count():
    cfg = dict(
        collector_env_num=8,
        episode_length=10,
        curriculum_enabled=True,
        start_distribution="curriculum",
        curriculum_total_env_steps=40000,
        curriculum_total_episodes=4,
    )
    collector_cfgs = SumToThreeEnv.create_collector_env_cfg(cfg)
    assert all(item["curriculum_total_episodes"] == 4 for item in collector_cfgs)


def test_evaluator_config_forces_binary_canonical():
    cfg = dict(
        evaluator_env_num=2,
        reward_algorithm="event_aligned",
        start_distribution="curriculum",
        curriculum_enabled=True,
    )
    evaluator_cfgs = SumToThreeEnv.create_evaluator_env_cfg(cfg)
    assert len(evaluator_cfgs) == 2
    for evaluator_cfg in evaluator_cfgs:
        assert evaluator_cfg["reward_algorithm"] == "binary"
        assert evaluator_cfg["start_distribution"] == "canonical"
        assert evaluator_cfg["curriculum_enabled"] is False
        assert evaluator_cfg["env_role"] == "evaluator"

    external_cfg = dict(
        evaluator_env_num=1,
        reward_algorithm="event_aligned",
        start_distribution="broad",
        external_evaluation=True,
    )
    external_evaluator = SumToThreeEnv.create_evaluator_env_cfg(external_cfg)[0]
    assert external_evaluator["reward_algorithm"] == "binary"
    assert external_evaluator["start_distribution"] == "broad"


def test_episode_info_contains_binary_diagnostics(env):
    env.seed(11, dynamic_seed=False)
    env.reset()
    timestep = env.step(_random_normalized_action())
    assert timestep.info == {}
    for _ in range(env.cfg.episode_length - 1):
        timestep = env.step(_random_normalized_action())

    assert timestep.done
    assert timestep.info["eval_episode_length"] == env.cfg.episode_length
    assert timestep.info["binary_episode_return"] == timestep.info["sparse_success_count"]
    assert timestep.info["reward_algorithm"] == "binary"
    assert timestep.info["start_distribution"] == "canonical"
    assert sum(timestep.info["cushion_count_histogram"].values()) == env.cfg.episode_length
