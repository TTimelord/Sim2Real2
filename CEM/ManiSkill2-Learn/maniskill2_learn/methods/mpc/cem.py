"""
Cross entropy method and it extensions, we use the tricks in iCEM by default.

MPPI: Model Predictive Path Integral Control using Covariance Variable Importance Sampling
    https://arxiv.org/pdf/1509.01149.pdf

iCEM: Sample-efficient Cross-Entropy Method for Real-time Planning
    https://arxiv.org/abs/2008.06389

PaETS: Variational Inference MPC for Bayesian Model-based Reinforcement Learning
    https://arxiv.org/abs/1907.04202

"""
from random import sample
from executing.executing import NodeFinder
import numpy as np
from contextlib import contextmanager
from scipy.special import softmax

from maniskill2_learn.schedulers import build_scheduler
from maniskill2_learn.utils.data import to_np
from maniskill2_learn.utils.data.dict_array import GDict
from maniskill2_learn.utils.math import trunc_normal
from maniskill2_learn.utils.meta import get_logger
from ..builder import MPC


@MPC.register_module()
class CEMOptimizer:
    """
    CEM is doing maximization here.
    """

    def __init__(
        self,
        eval_function,
        n_iter,
        population,
        elite,
        lr=1.0,
        bound=None,
        use_log=True,
        iscost=1,
        use_softmax=False,
        temperature=1.0,
        add_histroy_elites=False,
        **kwargs,
    ):
        self.eval_function = eval_function
        self.logger = get_logger(name="CEM", with_stream=use_log)

        self.n_iter = n_iter
        self.population = population
        self.elite = elite
        self.iscost = iscost
        self.add_histroy_elites = add_histroy_elites

        self.lr = lr
        self.temperature = temperature
        self.use_softmax = use_softmax
        self.use_trunc_normal = bound is not None

        self.lower_bound = to_np(bound[0], dtype="float32") if bound is not None else -np.inf
        self.upper_bound = to_np(bound[1], dtype="float32") if bound is not None else np.inf

    def get_params(self):
        return dict(population=self.population, elite=self.elite)

    def update_params(self, **kwargs):
        self.__dict__.update(kwargs)

    def __call__(self, state, mean=None, std=None):
        x_shape = (self.population,) + tuple(mean.shape)
        histroy_elites = None

        for idx in range(self.n_iter):
            ld = mean - self.lower_bound
            rd = self.upper_bound - mean
            _std = np.minimum(np.abs(np.minimum(ld, rd) / 2), std)

            if self.use_trunc_normal:
                noise = trunc_normal(x_shape)
            else:
                noise = np.random.randn(*x_shape)
            samples = noise * _std + mean
            reward, reward_per_step = self.eval_function(state, samples)

            all_infos = samples, reward, reward_per_step
            if histroy_elites is not None and self.add_histroy_elites:
                # Trick in iCEM, we need to utilize the history information and gaurantee the reward is increasing
                all_infos = [np.concatenate([a, b], axis=0) for a, b in zip(histroy_elites, all_infos)]
                samples, reward, reward_per_step = all_infos

            assert samples.ndim == 3 and reward.ndim == 1 and reward_per_step.ndim == 2, f"{samples.shape}, {reward.shape}, {reward_per_step.shape}"
            valid_sign = reward == reward
            reward = reward[valid_sign]
            reward_per_step = reward_per_step[valid_sign]
            samples = samples[valid_sign]
            optimal = samples[reward.argmax()]  # May be not stable

            if self.add_histroy_elites:
                topk_idx = np.argpartition(-reward, self.elite, axis=0)[: self.elite]
                histroy_elites = [_[topk_idx] for _ in all_infos]

            self.logger.info(
                f"CEM round {idx + 1}, action std: {std.max():.3f}, 1-setp-rew: {reward_per_step[:, 0].max() * self.iscost:.3f}, multi-step-avg-rew: {reward.max() * self.iscost:.3f}"
            )

            if self.use_softmax:
                exp_weight = softmax(self.temperature * reward, axis=0).reshape(*[-1] + [1 for _ in range(samples.ndim - 1)])
                elite_mean = (exp_weight * samples).sum(dim=0)
                elite_var = ((samples - elite_mean) ** 2 * exp_weight).sum(dim=0)
            else:
                topk_idx = np.argpartition(-reward, self.elite, axis=0)[: self.elite]
                elite = np.take(samples, topk_idx, axis=0)
                elite_mean = elite.mean(axis=0)
                elite_var = elite.std(axis=0) ** 2

            mean = mean * (1 - self.lr) + elite_mean * self.lr
            std = (std**2 * (1 - self.lr) + elite_var * self.lr) ** 0.5
        return elite_mean


@MPC.register_module()
class CEM:
    def __init__(self, rollout, env_params, cem_cfg, scheduler_config, horizon, action_horizon=1, add_actions=True, std=None, use_log=True, **kwargs):
        self.logger = get_logger(name="CEM", with_stream=use_log)
        self.logger.warning(
            "Current CEM implementation will not conisder the done signal, "
            "please make sure the environment can run and provide correct reward after done!"
        )

        self.rollout = rollout
        action_space = env_params["action_space"]
        self.action_space = action_space

        if action_space is not None and action_space.is_bounded():
            cem_cfg["bound"] = [action_space.low, action_space.high]

        cem_cfg["eval_function"] = self.reward
        cem_cfg["use_log"] = use_log
        self.optimizer = CEMOptimizer(**cem_cfg)
        self.cem_cfg = cem_cfg

        self.scheduler = build_scheduler(scheduler_config)

        self.iter = 0
        self.current_actions, self.action_buffer = None, None

        self.horizon = horizon
        self.action_horizon = action_horizon
        self.add_actions = add_actions

        if std is None:
            self.std = (action_space.high - action_space.low) / 4
        else:
            self.std = action_space.low * 0 + std

    def reward(self, state, a):
        x = np.repeat(state, a.shape[0], axis=0)
        ret = self.rollout.step_states_actions(x, a)[..., 0]
        return ret.mean(-1), ret

    def init_actions(self, horizon):
        return np.array([(self.action_space.high + self.action_space.low) * 0.5 for _ in range(horizon)], np.float32)

    def update_hyper(self):
        new_params = self.scheduler.get(self.cem_cfg, self.iter)
        self.optimizer.update_params(**new_params)

    def reset(self, **kwargs):
        # random sample may be not good
        self.rollout.reset(**kwargs)
        self.current_actions = self.init_actions(self.horizon)
        self.iter = 0
        self.action_buffer = None

    @contextmanager
    def no_sync(self, *args, **kwargs):
        yield

    def __call__(self, state, **kwargs):
        assert state.ndim == 1 or (state.ndim == 2 and state.shape[0] == 1)  # CEM only works for single environment.
        if self.action_buffer is not None and self.action_buffer.shape[0] > 0:
            action, self.action_buffer = self.action_buffer[:1], self.action_buffer[1:]
            self.iter += 1
            kwargs = dict(kwargs)
            if kwargs.get("rnn_mode", "base") != "base":
                return action, None
            else:
                return action
        self.update_hyper()
        self.current_actions = self.optimizer(state, self.current_actions, self.std)
        results = np.split(
            self.current_actions,
            [
                self.action_horizon,
            ],
            axis=0,
        )
        self.action_buffer, self.current_actions = results
        if self.add_actions:
            self.current_actions = np.concatenate([self.current_actions, self.current_actions[-1:]], axis=0)
        return self.__call__(state, **kwargs)
