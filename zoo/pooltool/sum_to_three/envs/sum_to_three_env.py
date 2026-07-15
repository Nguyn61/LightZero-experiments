"""
Overview:
    Implementation of a learning environment for the simple billiards game, sum-to-three.
    The game is played on a table with no pockets.
    There are 2 balls: a cue ball and an object ball
    The player must hit the object ball with the cue ball
    The player scores a point if the number of times a ball hits a cushion is 3
Mode:
    - ``self_play_mode``: In ``self_play_mode``, there is only one player, who takes 10 \
        shots. Their final score is the number of points they achieve.
    - ``play_with_bot_mode``: (**NOT YET IMPLEMENTED**) In ``play_with_bot_mode`` there are two players, \
        a learning agent and a bot. The game ends when either player achieves 5 points.
Bot:
    - MCTSBot: (**NOT YET IMPLEMENTED**) A bot which take action through a Monte Carlo Tree Search, which \
        has a high performance.
    - RuleBot: (**NOT YET IMPLEMENTED**) A bot which takes actions according to some simple heuristics, \
        which has a low performance.
Observation Space:
    The observation in this environment is a dictionary with three elements.
    - observation (:obj:`array`): A continuous 1D array holding the x- and y- coordinates of the cue ball \
        and the object ball. It has the following entries: ``[x_cue, y_cue, x_obj, y_obj]``. ``x_cue`` and \
        ``y_cue`` are the 2D coordinates of the cue ball, and ``x_obj`` and ``y_obj`` are 2D coordinates of \
        the object ball. x-coordinates can be between ``R`` and ``w-R``, where ``R`` is the ball radius and \
        ``w`` is the width of the table. Similarly, y-coordinates can be between ``R`` and ``l-R``, where \
        ``l`` is the length of the table.
    - action_mask (:obj:`None`): No actions are masked, so ``None`` is used here.
    - to_play (:obj:`None`): (**NOT YET IMPLEMENTED**) For ``self_play_mode``, this is
        set to -1. For ``play_with_bot_mode``, this indicates the player that needs to take an action in the \
        current state.
Action Space:
    A continuous length-2 array. The first element is ``V0``, the speed of the cue stick. The second element \
    is the ``cut_angle``, which is the angle that the cue ball hits the object ball with. A cut angle of 0 is \
    a head-on collision, a cut angle of -89 is a very slight graze on the left side of the object ball, and a \
    cut angle of 89 is a very slight graze on the right side of the object ball.
Reward Space:
    For the ``self_play_mode``, intermediate rewards of 1.0 are returned for each step where the player earns a point.
    For the ``play_with_bot_mode``, (**NOT YET IMPLEMENTED**)...
"""

from __future__ import annotations

import copy
import gc
import math

from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Dict, Optional, Tuple
import numpy as np
from ding.envs import BaseEnvTimestep
from ding.utils import ENV_REGISTRY
from easydict import EasyDict
from numpy.typing import NDArray
from zoo.pooltool.datatypes import (
    Bounds,
    ObservationDict,
    PoolToolEnv,
    PoolToolSimulator,
    Spaces,
    State,
)
from gym import spaces

import pooltool as pt
from zoo.pooltool.image_representation import PygameRenderer, RenderConfig
from zoo.pooltool.sum_to_three.envs.utils import (
    ObservationType,
    coordinate_observation_array,
    image_observation_array,
    get_image_obs_space,
    get_coordinate_obs_space,
    get_reward_space,
    get_shot_outcome,
    reward_from_outcome,
)


def get_action_space(V0: Bounds, angle: Bounds) -> spaces.Box:
    """
    Overview:
        Given the action bounds, return the action space.
    """
    return spaces.Box(
        low=np.array([V0.low, angle.low], dtype=np.float32),
        high=np.array([V0.high, angle.high], dtype=np.float32),
        shape=(2,),
        dtype=np.float32,
    )


class StartDistribution(Enum):
    CANONICAL = "canonical"
    LOCAL = "local"
    BROAD = "broad"
    CURRICULUM = "curriculum"


def _canonical_positions(system: pt.System) -> Tuple[np.ndarray, np.ndarray]:
    R = system.balls["cue"].params.R
    cue_pos = np.array([system.table.w / 2, system.table.l / 4, R], dtype=np.float64)
    object_pos = np.array([system.table.w / 2, system.table.l * 3 / 4, R], dtype=np.float64)
    return cue_pos, object_pos


def _valid_positions(
    cue_pos: np.ndarray,
    object_pos: np.ndarray,
    system: pt.System,
    separation_margin: float,
) -> bool:
    R = system.balls["cue"].params.R
    within_table = (
        R <= cue_pos[0] <= system.table.w - R
        and R <= cue_pos[1] <= system.table.l - R
        and R <= object_pos[0] <= system.table.w - R
        and R <= object_pos[1] <= system.table.l - R
    )
    separation = np.linalg.norm(cue_pos[:2] - object_pos[:2])
    return within_table and separation >= 2 * R + separation_margin


def sample_initial_positions(
    system: pt.System,
    distribution: StartDistribution,
    rng: np.random.Generator,
    local_perturbation_fraction: float = 0.08,
    separation_margin: float = 0.005,
    max_attempts: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample valid ball positions for one of the research start distributions."""
    canonical_cue, canonical_object = _canonical_positions(system)
    if distribution == StartDistribution.CANONICAL:
        return canonical_cue, canonical_object
    if distribution == StartDistribution.CURRICULUM:
        raise ValueError("CURRICULUM must be resolved to canonical, local, or broad before sampling")

    R = system.balls["cue"].params.R
    usable_width = system.table.w - 2 * R
    usable_length = system.table.l - 2 * R

    for _ in range(max_attempts):
        if distribution == StartDistribution.LOCAL:
            scale = np.array(
                [local_perturbation_fraction * usable_width, local_perturbation_fraction * usable_length]
            )
            cue_xy = canonical_cue[:2] + rng.uniform(-scale, scale)
            object_xy = canonical_object[:2] + rng.uniform(-scale, scale)
        elif distribution == StartDistribution.BROAD:
            cue_xy = rng.uniform([R, R], [system.table.w - R, system.table.l - R])
            object_xy = rng.uniform([R, R], [system.table.w - R, system.table.l - R])
        else:
            raise ValueError(f"Unhandled start distribution: {distribution}")

        cue_pos = np.array([cue_xy[0], cue_xy[1], R], dtype=np.float64)
        object_pos = np.array([object_xy[0], object_xy[1], R], dtype=np.float64)
        if _valid_positions(cue_pos, object_pos, system, separation_margin):
            return cue_pos, object_pos

    raise RuntimeError(
        f"Could not sample valid {distribution.value} positions after {max_attempts} attempts"
    )


def _set_initial_positions(
    system: pt.System,
    random_pos: bool = False,
    *,
    distribution: Optional[StartDistribution] = None,
    rng: Optional[np.random.Generator] = None,
    local_perturbation_fraction: float = 0.08,
    separation_margin: float = 0.005,
    max_attempts: int = 100,
) -> None:
    if distribution is None:
        distribution = StartDistribution.BROAD if random_pos else StartDistribution.CANONICAL
    if rng is None:
        legacy_seed = int(np.random.randint(0, np.iinfo(np.uint32).max))
        rng = np.random.default_rng(legacy_seed)

    cue_pos, object_pos = sample_initial_positions(
        system,
        distribution,
        rng,
        local_perturbation_fraction=local_perturbation_fraction,
        separation_margin=separation_margin,
        max_attempts=max_attempts,
    )
    system.balls["cue"].state.rvw[0] = cue_pos
    system.balls["object"].state.rvw[0] = object_pos


def _set_initial_cue_state(system: pt.System) -> None:
    system.cue.set_state(
        V0=0.0,
        phi=0.0,
        theta=0.0,
        a=0.0,
        b=0.0,
    )


def create_initial_state(random_pos: bool) -> State:
    """
    Overview:
        Creates a ready-to-play state.
    Arguments:
        - random_pos: If ``False``, initial ball positions are set to the starting \
            configuration of the game (with the cue ball on one side of the table and the \
            object ball on the other side). If ``True``, the ball positions are randomized.
    Returns:
        - state (:obj:`State`): The ready-to-play state. The game is setup to be single \
            player with perpetual play (no win condition). The cue stick parameters have not \
            yet been set.
    """
    gametype = pt.GameType.SUMTOTHREE
    players = [pt.Player("Player 1")]

    game = pt.get_ruleset(gametype)(
        players=players,
        win_condition=-1,  # type: ignore
    )

    system = pt.System(
        cue=pt.Cue.default(),
        table=(table := pt.Table.from_game_type(gametype)),
        balls=pt.get_rack(gametype, table),
    )

    _set_initial_positions(system, random_pos)
    _set_initial_cue_state(system)

    return State(system, game)


@dataclass
class SumToThreeSimulator(PoolToolSimulator):
    """
    Overview:
        Manages the simulation state for simulating actions and retrieving subsequent \
        observations.
    """

    observation_type: ObservationType
    renderer: Optional[PygameRenderer] = None

    def set_action(self, action: NDArray[np.float32]) -> None:
        """
        Overview:
            Sets the cue stick state for a 2-parameter action.
        Arguments:
            - action (:obj:`NDArray[np.float32]`): A length-2 array, where the first \
                parameter is the speed of the cue stick (in m/s), and the second is \
                the cut angle (i.e., the angle that the cue ball hits the object \
                ball with) (in degrees). Spin and strike elevation are set to 0.
        """
        self.state.system.cue.set_state(
            V0=action[0],
            phi=pt.aim.at_ball(self.state.system, "object", cut=action[1]),
            theta=0.0,
            a=0.0,
            b=0.0,
        )

    def observation_array(self) -> NDArray[np.float32]:
        """
        Overview:
            Returns an observation array of the current state.
        Returns:
            - observation (:obj:`NDArray[np.float32]`): The observation array. For
                details, see the docstrings of the delegate functions.
        """
        if self.observation_type == ObservationType.COORDINATE:
            return coordinate_observation_array(self.state)
        elif self.observation_type == ObservationType.IMAGE:
            assert self.renderer is not None
            assert self.renderer.state is self.state
            return image_observation_array(self.renderer)

        raise ValueError(f"Unhandled Enum member '{self.observation_type}'")

    def reset(self) -> None:
        if len(self.state.game.players) == 1:
            self.reset_single_player_env()
        else:
            raise NotImplementedError()

    def reset_single_player_env(self) -> None:
        """Return the passed environment, resetting things to an initial state"""
        del self.state.game
        self.state.game = pt.get_ruleset(pt.GameType.SUMTOTHREE)(
            players=[pt.Player("Player 1")],
            win_condition=-1,  # type: ignore
        )

        self.state.system.reset_history()
        self.state.system.stop_balls()

        # Set ball positions at the starting place
        _set_initial_positions(self.state.system, random_pos=False)
        _set_initial_cue_state(self.state.system)

        assert self.state.system.balls["cue"].state.s == pt.constants.stationary
        assert self.state.system.balls["object"].state.s == pt.constants.stationary
        assert not np.isnan(self.state.system.balls["cue"].state.rvw).any()
        assert not np.isnan(self.state.system.balls["object"].state.rvw).any()


@dataclass
class EpisodicTrackedStats:
    eval_episode_length: int = 0
    eval_episode_return: float = 0.0
    binary_episode_return: float = 0.0
    sparse_success_count: int = 0
    contact_count: int = 0
    cushion_count_histogram: Dict[str, int] = field(default_factory=dict)
    start_distribution: str = StartDistribution.CANONICAL.value
    reward_algorithm: str = "binary"


@ENV_REGISTRY.register("pooltool_sumtothree")
class SumToThreeEnv(PoolToolEnv):
    config = dict(
        env_name="PoolTool-SumToThree",
        env_type="not_board_games",
        episode_length=10,
        reward_algorithm="binary",
        action_V0_low=0.3,
        action_V0_high=3.0,
        action_angle_low=-70,
        action_angle_high=70,
        raw_observation=False,
        start_distribution="canonical",
        curriculum_enabled=False,
        curriculum_total_env_steps=40000,
        curriculum_total_episodes=None,
        curriculum_canonical_fraction=0.25,
        curriculum_local_fraction=0.40,
        local_perturbation_fraction=0.08,
        start_separation_margin=0.005,
        start_sampling_max_attempts=100,
        emit_step_diagnostics=False,
        env_role="default",
    )

    def __repr__(self) -> str:
        return "SumToThreeEnv"

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> list:
        cfg = copy.deepcopy(cfg)
        collector_env_num = cfg.pop("collector_env_num")
        cfg["env_role"] = "collector"
        if (
            cfg.get("curriculum_total_episodes") is None
            and (
                cfg.get("curriculum_enabled")
                or cfg.get("start_distribution") == StartDistribution.CURRICULUM.value
            )
        ):
            total_env_steps = int(cfg.get("curriculum_total_env_steps", 40000))
            episode_length = int(cfg.get("episode_length", SumToThreeEnv.config["episode_length"]))
            cfg["curriculum_total_episodes"] = max(
                1, math.ceil(total_env_steps / (collector_env_num * episode_length))
            )
        return [copy.deepcopy(cfg) for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> list:
        cfg = copy.deepcopy(cfg)
        evaluator_env_num = cfg.pop("evaluator_env_num")
        start_distribution = (
            cfg.get("start_distribution", StartDistribution.CANONICAL.value)
            if cfg.get("external_evaluation", False)
            else StartDistribution.CANONICAL.value
        )
        cfg.update(
            env_role="evaluator",
            reward_algorithm="binary",
            start_distribution=start_distribution,
            curriculum_enabled=False,
        )
        return [copy.deepcopy(cfg) for _ in range(evaluator_env_num)]

    def __init__(self, cfg: EasyDict) -> None:
        self.cfg = cfg
        self.raw_observation = cfg.get("raw_observation", False)

        # Structure the action bounds
        self.action_bounds = {
            "V0": Bounds(
                low=self.cfg.action_V0_low,
                high=self.cfg.action_V0_high,
            ),
            "angle": Bounds(
                low=self.cfg.action_angle_low,
                high=self.cfg.action_angle_high,
            ),
        }

        try:
            self.observation_type = ObservationType(self.cfg.observation_type)
        except AttributeError:
            available = [
                member.value for member in ObservationType.__members__.values()
            ]
            raise ValueError(f"Must set 'observation_type' to one of {available}.")
        except ValueError:
            available = [
                member.value for member in ObservationType.__members__.values()
            ]
            raise ValueError(f"'observation_type' must be one of {available}.")

        if self.observation_type == ObservationType.IMAGE:
            if "render_config_path" in self.cfg:
                self.render_config = RenderConfig.from_json(self.cfg.render_config_path)
            else:
                self.render_config = RenderConfig.default()

        self._init_flag = False
        self._rng = np.random.default_rng()
        self._reset_count = 0
        self._current_start_distribution = StartDistribution.CANONICAL
        self._tracked_stats = EpisodicTrackedStats()
        self._env: SumToThreeSimulator

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        super().seed(seed, dynamic_seed)
        self._rng = np.random.default_rng(seed)

    def _resolve_start_distribution(self) -> StartDistribution:
        configured = StartDistribution(self.cfg.get("start_distribution", "canonical"))
        if not self.cfg.get("curriculum_enabled", False) and configured != StartDistribution.CURRICULUM:
            return configured

        total_episodes = self.cfg.get("curriculum_total_episodes")
        if total_episodes is None:
            total_episodes = max(
                1,
                math.ceil(
                    int(self.cfg.get("curriculum_total_env_steps", 40000))
                    / int(self.cfg.get("episode_length", self.config["episode_length"]))
                ),
            )
        progress = min(self._reset_count / max(int(total_episodes), 1), 1.0)
        canonical_fraction = float(self.cfg.get("curriculum_canonical_fraction", 0.25))
        local_fraction = float(self.cfg.get("curriculum_local_fraction", 0.40))
        if progress < canonical_fraction:
            return StartDistribution.CANONICAL
        if progress < canonical_fraction + local_fraction:
            return StartDistribution.LOCAL
        return StartDistribution.BROAD

    def _apply_start_distribution(self) -> None:
        self._current_start_distribution = self._resolve_start_distribution()
        _set_initial_positions(
            self._env.state.system,
            distribution=self._current_start_distribution,
            rng=self._rng,
            local_perturbation_fraction=float(self.cfg.get("local_perturbation_fraction", 0.08)),
            separation_margin=float(self.cfg.get("start_separation_margin", 0.005)),
            max_attempts=int(self.cfg.get("start_sampling_max_attempts", 100)),
        )
        _set_initial_cue_state(self._env.state.system)
        self._reset_count += 1

    def close(self) -> None:
        if self._env.renderer is not None:
            self._env.renderer.close()

        # Probably not necessary
        for ball in self._env.state.system.balls.values():
            del ball.state
            del ball.history
            del ball.history_cts
            del ball
        for pocket in self._env.state.system.table.pockets.values():
            del pocket
        for cushion in self._env.state.system.table.cushion_segments.linear.values():
            del cushion
        for cushion in self._env.state.system.table.cushion_segments.circular.values():
            del cushion
        del self._env.state.system.table
        del self._env.state.system.cue
        del self._env.state.system
        del self._env.state.game
        del self._env
        gc.collect()

        self._init_flag = False

    def reset(self) -> ObservationDict:
        if not self._init_flag:
            state = create_initial_state(random_pos=False)
            renderer = None

            if self.observation_type == ObservationType.COORDINATE:
                observation_space = get_coordinate_obs_space(state.system)
            elif self.observation_type == ObservationType.IMAGE:
                # setup renderer
                observation_space = get_image_obs_space(self.render_config)
                renderer = PygameRenderer.build(state.system.table, self.render_config)
                renderer.set_state(state)
                renderer.init()
            else:
                raise ValueError(f"Unhandled Enum member '{self.observation_type}'")

            action_space = get_action_space(
                self.action_bounds["V0"],
                self.action_bounds["angle"],
            )
            reward_space = get_reward_space(
                self.cfg.reward_algorithm,
            )
            spaces = Spaces(
                observation_space,
                action_space,
                reward_space,
            )

            # Create the environment
            self._env = SumToThreeSimulator(
                state,
                spaces,
                observation_type=self.observation_type,
                renderer=renderer,
            )

            self._init_flag = True
        else:
            self._env.reset()

        self.manage_seeds()
        self._apply_start_distribution()
        self._tracked_stats = EpisodicTrackedStats(
            start_distribution=self._current_start_distribution.value,
            reward_algorithm=self.cfg.reward_algorithm,
        )

        self._observation_space = self._env.spaces.observation
        self._action_space = self._env.spaces.action
        self._reward_space = self._env.spaces.reward

        if self.raw_observation:
            return self._env.observation_raw()
        else:
            return self._env.observation()

    def step(self, action: NDArray[np.float32]) -> BaseEnvTimestep:
        self._env.set_action(self._env.scale_action(action))
        self._env.simulate()

        outcome = get_shot_outcome(self._env.state)
        rew = reward_from_outcome(self.cfg.reward_algorithm, outcome)
        binary_rew = float(outcome.sparse_success)

        self._tracked_stats.eval_episode_length += 1
        self._tracked_stats.eval_episode_return += rew
        self._tracked_stats.binary_episode_return += binary_rew
        self._tracked_stats.sparse_success_count += int(outcome.sparse_success)
        self._tracked_stats.contact_count += int(outcome.contacted_object)
        cushion_key = str(outcome.linear_cushion_count)
        self._tracked_stats.cushion_count_histogram[cushion_key] = (
            self._tracked_stats.cushion_count_histogram.get(cushion_key, 0) + 1
        )

        done = self._tracked_stats.eval_episode_length == self.cfg.episode_length
        info = {}
        if self.cfg.get("emit_step_diagnostics", False) or done:
            info.update(
                shot_outcome=asdict(outcome),
                binary_reward=binary_rew,
                start_distribution=self._current_start_distribution.value,
                reward_algorithm=self.cfg.reward_algorithm,
            )
        if done:
            info.update(asdict(self._tracked_stats))
            info["episode_info"] = {
                "binary_episode_return": self._tracked_stats.binary_episode_return,
                "sparse_success_count": self._tracked_stats.sparse_success_count,
                "contact_count": self._tracked_stats.contact_count,
            }

        if self.raw_observation:
            return BaseEnvTimestep(
                obs=self._env.observation_raw(),
                reward=np.array([rew], dtype=np.float32),
                done=done,
                info=info,
            )
        else:
            return BaseEnvTimestep(
                obs=self._env.observation(),
                reward=np.array([rew], dtype=np.float32),
                done=done,
                info=info,
            )
