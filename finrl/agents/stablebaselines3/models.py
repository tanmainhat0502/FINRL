# DRL models from Stable Baselines 3
from __future__ import annotations

import statistics
import time

import numpy as np
import pandas as pd

from finrl.agents.stablebaselines3.stable_baselines3.ppo_recurrent.ppo_recurrent import RecurrentPPO
from stable_baselines3.a2c import A2C
from stable_baselines3.ddpg import DDPG
from stable_baselines3.ppo import PPO
from stable_baselines3.sac import SAC
from stable_baselines3.td3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from finrl.agents.stablebaselines3.stable_baselines3.common.recurrent.policies import RecurrentActorCriticPolicy
from finrl import config
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import data_split

MODELS = {"a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO, "re_ppo": RecurrentPPO}
MODEL_KWARGS = {x: config.__dict__[f"{x.upper()}_PARAMS"] for x in MODELS.keys()}

NOISE = {
    "normal": NormalActionNoise,
    "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,
}


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        try:
            self.logger.record(key="train/reward", value=self.locals["rewards"][0])

        except BaseException as error:
            try:
                self.logger.record(key="train/reward", value=self.locals["reward"][0])

            except BaseException as inner_error:
                # Handle the case where neither "rewards" nor "reward" is found
                self.logger.record(key="train/reward", value=None)
                # Print the original error and the inner error for debugging
                print("Original Error:", error)
                print("Inner Error:", inner_error)
        return True

    def _on_rollout_end(self) -> bool:
        try:
            rollout_buffer_rewards = self.locals["rollout_buffer"].rewards.flatten()
            self.logger.record(
                key="train/reward_min", value=min(rollout_buffer_rewards)
            )
            self.logger.record(
                key="train/reward_mean", value=statistics.mean(rollout_buffer_rewards)
            )
            self.logger.record(
                key="train/reward_max", value=max(rollout_buffer_rewards)
            )
        except BaseException as error:
            # Handle the case where "rewards" is not found
            self.logger.record(key="train/reward_min", value=None)
            self.logger.record(key="train/reward_mean", value=None)
            self.logger.record(key="train/reward_max", value=None)
            print("Logging Error:", error)
        return True


# class DRLAgent:
#     """Provides implementations for DRL algorithms

#     Attributes
#     ----------
#         env: gym environment class
#             user-defined class

#     Methods
#     -------
#         get_model()
#             setup DRL algorithms
#         train_model()
#             train DRL algorithms in a train dataset
#             and output the trained model
#         DRL_prediction()
#             make a prediction in a test dataset and get results
#     """

#     def __init__(self, env):
#         self.env = env

#     def get_model(
#         self,
#         model_name,
#         policy=None,
#         model_kwargs=None,
#         verbose=1,
#         seed=None,
#     ):
#         if model_name not in MODELS:
#             raise ValueError(
#                 f"Model '{model_name}' not found in MODELS."
#             )

#         if model_kwargs is None:
#             model_kwargs = MODEL_KWARGS[model_name]

#         if "action_noise" in model_kwargs:
#             n_actions = self.env.action_space.shape[-1]
#             model_kwargs["action_noise"] = NOISE[model_kwargs["action_noise"]](
#                 mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
#             )
#         print(model_kwargs)

#         # Xác định policy dựa trên model_name
#         if model_name == "re_ppo":
#             policy = RecurrentActorCriticPolicy
#         else:
#             policy = 'MlpPolicy' 

#         return MODELS[model_name](
#             policy=policy,
#             env=self.env,
#             verbose=verbose,
#             policy_kwargs=policy_kwargs,
#             seed=seed,
#             **model_kwargs,
#         )

#     @staticmethod
#     def train_model(
#         model,
#         tb_log_name,
#         total_timesteps=5000,
#         callbacks: Type[BaseCallback] = None,
#     ):  # this function is static method, so it can be called without creating an instance of the class
#         model = model.learn(
#             total_timesteps=total_timesteps,
#             tb_log_name=tb_log_name,
#             callback=(
#                 CallbackList(
#                     [TensorboardCallback()] + [callback for callback in callbacks]
#                 )
#                 if callbacks is not None
#                 else TensorboardCallback()
#             ),
#         )
#         return model

#     @staticmethod
#     def DRL_prediction(model, environment, deterministic=True):
#         """make a prediction and get results"""
#         test_env, test_obs = environment.get_sb_env()
#         account_memory = None  # This help avoid unnecessary list creation
#         actions_memory = None  # optimize memory consumption
#         # state_memory=[] #add memory pool to store states

#         test_env.reset()
#         max_steps = len(environment.df.index.unique()) - 1

#         for i in range(len(environment.df.index.unique())):
#             action, _states = model.predict(test_obs, deterministic=deterministic)
#             # account_memory = test_env.env_method(method_name="save_asset_memory")
#             # actions_memory = test_env.env_method(method_name="save_action_memory")
#             test_obs, rewards, dones, info = test_env.step(action)

#             if (
#                 i == max_steps - 1
#             ):  # more descriptive condition for early termination to clarify the logic
#                 account_memory = test_env.env_method(method_name="save_asset_memory")
#                 actions_memory = test_env.env_method(method_name="save_action_memory")
#             # add current state to state memory
#             # state_memory=test_env.env_method(method_name="save_state_memory")

#             if dones[0]:
#                 print("hit end!")
#                 break
#         return account_memory[0], actions_memory[0]

#     @staticmethod
#     def DRL_prediction_load_from_file(model_name, environment, cwd, deterministic=True):
#         if model_name not in MODELS:
#             raise ValueError(
#                 f"Model '{model_name}' not found in MODELS."
#             )  # this is more informative than NotImplementedError("NotImplementedError")
#         try:
#             # load agent
#             model = MODELS[model_name].load(cwd)
#             print("Successfully load model", cwd)
#         except BaseException as error:
#             raise ValueError(f"Failed to load agent. Error: {str(error)}") from error

#         # test on the testing env
#         state = environment.reset()
#         episode_returns = []  # the cumulative_return / initial_account
#         episode_total_assets = [environment.initial_total_asset]
#         done = False
#         while not done:
#             action = model.predict(state, deterministic=deterministic)[0]
#             state, reward, done, _ = environment.step(action)

#             total_asset = (
#                 environment.amount
#                 + (environment.price_ary[environment.day] * environment.stocks).sum()
#             )
#             episode_total_assets.append(total_asset)
#             episode_return = total_asset / environment.initial_total_asset
#             episode_returns.append(episode_return)

#         print("episode_return", episode_return)
#         print("Test Finished!")
#         return episode_total_assets



class DRLAgent:
    """Provides implementations for DRL algorithms

    Attributes
    ----------
        env: gym environment class
            user-defined class

    Methods
    -------
        get_model()
            setup DRL algorithms
        train_model()
            train DRL algorithms in a train dataset
            and output the trained model
        DRL_prediction()
            make a prediction in a test dataset and get results
    """

    def __init__(self, env):
        self.env = env

    def get_model(
        self,
        model_name,
        policy=None,
        model_kwargs=None,
        verbose=1,
        seed=None,
    ):
        if model_name not in MODELS:
            raise ValueError(
                f"Model '{model_name}' not found in MODELS."
            )

        if model_kwargs is None:
            model_kwargs = MODEL_KWARGS[model_name]
        else:
            # Loại bỏ verbose và seed khỏi model_kwargs để tránh xung đột
            if "verbose" in model_kwargs:
                del model_kwargs["verbose"]
            if "seed" in model_kwargs:
                del model_kwargs["seed"]

        if "action_noise" in model_kwargs:
            n_actions = self.env.action_space.shape[-1]
            model_kwargs["action_noise"] = NOISE[model_kwargs["action_noise"]](
                mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
            )
        print(model_kwargs)

        # Xác định policy dựa trên model_name
        if model_name == "re_ppo":
            policy = RecurrentActorCriticPolicy
            if "policy_kwargs" not in model_kwargs:
                model_kwargs["policy_kwargs"] = {}
            model_kwargs["policy_kwargs"]["context_length"] = 31  # Thêm context_length
        else:
            policy = 'MlpPolicy'

        return MODELS[model_name](
            policy=policy,
            env=self.env,
            verbose=verbose,
            seed=seed,  # Sử dụng seed từ tham số hàm
            **model_kwargs,
        )

    @staticmethod
    def train_model(
        model,
        tb_log_name,
        total_timesteps=5000,
        callbacks: Type[BaseCallback] = None,
    ):  # this function is static method, so it can be called without creating an instance of the class
        model = model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name,
            callback=(
                CallbackList(
                    [TensorboardCallback()] + [callback for callback in callbacks]
                )
                if callbacks is not None
                else TensorboardCallback()
            ),
        )
        return model

    @staticmethod
    def DRL_prediction(model, environment, deterministic=True):
        """make a prediction and get results"""
        test_env, test_obs = environment.get_sb_env()
        account_memory = None  # This help avoid unnecessary list creation
        actions_memory = None  # optimize memory consumption
        # state_memory=[] #add memory pool to store states

        test_env.reset()
        max_steps = len(environment.df.index.unique()) - 1

        for i in range(len(environment.df.index.unique())):
            action, _states = model.predict(test_obs, deterministic=deterministic)
            # account_memory = test_env.env_method(method_name="save_asset_memory")
            # actions_memory = test_env.env_method(method_name="save_action_memory")
            test_obs, rewards, dones, info = test_env.step(action)

            if (
                i == max_steps - 1
            ):  # more descriptive condition for early termination to clarify the logic
                account_memory = test_env.env_method(method_name="save_asset_memory")
                actions_memory = test_env.env_method(method_name="save_action_memory")
            # add current state to state memory
            # state_memory=test_env.env_method(method_name="save_state_memory")

            if dones[0]:
                print("hit end!")
                break
        return account_memory[0], actions_memory[0]

    @staticmethod
    def DRL_prediction_load_from_file(model_name, environment, cwd, deterministic=True):
        if model_name not in MODELS:
            raise ValueError(
                f"Model '{model_name}' not found in MODELS."
            )  # this is more informative than NotImplementedError("NotImplementedError")
        try:
            # load agent
            model = MODELS[model_name].load(cwd)
            print("Successfully load model", cwd)
        except BaseException as error:
            raise ValueError(f"Failed to load agent. Error: {str(error)}") from error

        # test on the testing env
        state = environment.reset()
        episode_returns = []  # the cumulative_return / initial_account
        episode_total_assets = [environment.initial_total_asset]
        done = False
        while not done:
            action = model.predict(state, deterministic=deterministic)[0]
            state, reward, done, _ = environment.step(action)

            total_asset = (
                environment.amount
                + (environment.price_ary[environment.day] * environment.stocks).sum()
            )
            episode_total_assets.append(total_asset)
            episode_return = total_asset / environment.initial_total_asset
            episode_returns.append(episode_return)

        print("episode_return", episode_return)
        print("Test Finished!")
        return episode_total_assets



class DRLEnsembleAgent:
    @staticmethod
    def get_model(
        model_name,
        env,
        policy=None,
        policy_kwargs=None,
        model_kwargs=None,
        seed=None,
        verbose=1,
    ):
        if model_name not in MODELS:
            raise ValueError(
                f"Model '{model_name}' not found in MODELS."
            )

        if model_kwargs is None:
            temp_model_kwargs = MODEL_KWARGS[model_name].copy()
        else:
            temp_model_kwargs = model_kwargs.copy()

        # Xác định policy dựa trên model_name
        if model_name == "re_ppo": policy = RecurrentActorCriticPolicy 
        else:
            policy = "MlpPolicy"
        # Loại bỏ policy_kwargs nếu không phải re_ppo hoặc không hợp lệ
        if model_name != "re_ppo" and "policy_kwargs" in temp_model_kwargs:
            temp_model_kwargs.pop("policy_kwargs", None)

        if "action_noise" in temp_model_kwargs:
            n_actions = env.action_space.shape[-1]
            temp_model_kwargs["action_noise"] = NOISE[
                temp_model_kwargs["action_noise"]
            ](mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

        # Loại bỏ seed và verbose trực tiếp nếu đã có trong temp_model_kwargs
        if "seed" in temp_model_kwargs:
            seed = temp_model_kwargs.pop("seed")
        if "verbose" in temp_model_kwargs:
            verbose = temp_model_kwargs.pop("verbose")

        # Sử dụng policy_kwargs từ temp_model_kwargs cho re_ppo
        effective_policy_kwargs = temp_model_kwargs.pop("policy_kwargs", None) if model_name == "re_ppo" else None

        print(temp_model_kwargs)
        return MODELS[model_name](
            policy=policy,
            env=env,
            verbose=verbose,
            policy_kwargs=effective_policy_kwargs,
            seed=seed,
            **temp_model_kwargs,
        )

    @staticmethod
    def train_model(
        model,
        model_name,
        tb_log_name,
        iter_num,
        total_timesteps=5000,
        callbacks: Type[BaseCallback] = None,
    ):
        model = model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name,
            callback=(
                CallbackList(
                    [TensorboardCallback()] + [callback for callback in callbacks]
                )
                if callbacks is not None
                else TensorboardCallback()
            ),
        )
        model.save(
            f"{config.TRAINED_MODEL_DIR}/{model_name.upper()}_{total_timesteps // 1000}k_{iter_num}"
        )
        return model

    @staticmethod
    def get_validation_sharpe(iteration, model_name):
        """Calculate Sharpe ratio based on validation results"""
        df_total_value = pd.read_csv(
            f"results/account_value_validation_{model_name}_{iteration}.csv"
        )
        # If the agent did not make any transaction
        if df_total_value["daily_return"].var() == 0:
            if df_total_value["daily_return"].mean() > 0:
                return np.inf
            else:
                return 0.0
        else:
            return (
                (4**0.5)
                * df_total_value["daily_return"].mean()
                / df_total_value["daily_return"].std()
            )

    def __init__(
        self,
        df,
        train_period,
        val_test_period,
        rebalance_window,
        validation_window,
        stock_dim,
        hmax,
        initial_amount,
        buy_cost_pct,
        sell_cost_pct,
        reward_scaling,
        state_space,
        action_space,
        tech_indicator_list,
        print_verbosity,
    ):
        self.df = df
        self.train_period = train_period
        self.val_test_period = val_test_period

        self.unique_trade_date = df[
            (df.date > val_test_period[0]) & (df.date <= val_test_period[1])
        ].date.unique()
        self.rebalance_window = rebalance_window
        self.validation_window = validation_window

        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.print_verbosity = print_verbosity
        self.train_env = None  # defined in train_validation() function

    def DRL_validation(self, model, test_data, test_env, test_obs):
        """validation process"""
        for _ in range(len(test_data.index.unique())):
            action, _states = model.predict(test_obs)
            test_obs, rewards, dones, info = test_env.step(action)

    def DRL_prediction(
        self, model, name, last_state, iter_num, turbulence_threshold, initial
    ):
        """make a prediction based on trained model"""

        # trading env
        trade_data = data_split(
            self.df,
            start=self.unique_trade_date[iter_num - self.rebalance_window],
            end=self.unique_trade_date[iter_num],
        )
        trade_env = DummyVecEnv(
            [
                lambda: StockTradingEnv(
                    df=trade_data,
                    stock_dim=self.stock_dim,
                    hmax=self.hmax,
                    initial_amount=self.initial_amount,
                    num_stock_shares=[0] * self.stock_dim,
                    buy_cost_pct=[self.buy_cost_pct] * self.stock_dim,
                    sell_cost_pct=[self.sell_cost_pct] * self.stock_dim,
                    reward_scaling=self.reward_scaling,
                    state_space=self.state_space,
                    action_space=self.action_space,
                    tech_indicator_list=self.tech_indicator_list,
                    turbulence_threshold=turbulence_threshold,
                    initial=initial,
                    previous_state=last_state,
                    model_name=name,
                    mode="trade",
                    iteration=iter_num,
                    print_verbosity=self.print_verbosity,
                )
            ]
        )

        trade_obs = trade_env.reset()

        for i in range(len(trade_data.index.unique())):
            action, _states = model.predict(trade_obs)
            trade_obs, rewards, dones, info = trade_env.step(action)
            if i == (len(trade_data.index.unique()) - 2):
                # print(env_test.render())
                last_state = trade_env.envs[0].render()

        df_last_state = pd.DataFrame({"last_state": last_state})
        df_last_state.to_csv(f"results/last_state_{name}_{i}.csv", index=False)
        return last_state

    def _train_window(
        self,
        model_name,
        model_kwargs,
        sharpe_list,
        validation_start_date,
        validation_end_date,
        timesteps_dict,
        i,
        validation,
        turbulence_threshold,
    ):
        """
        Train the model for a single window.
        """
        if model_kwargs is None:
            return None, sharpe_list, -1

        print(f"======{model_name} Training========")
        policy = "RecurrentActorCriticPolicy" if model_name == "re_ppo" else "MlpPolicy"
        model = self.get_model(
            model_name, self.train_env, policy=policy, model_kwargs=model_kwargs
        )
        model = self.train_model(
            model,
            model_name,
            tb_log_name=f"{model_name}_{i}",
            iter_num=i,
            total_timesteps=timesteps_dict[model_name],
        )  # 100_000
        print(
            f"======{model_name} Validation from: ",
            validation_start_date,
            "to ",
            validation_end_date,
        )
        val_env = DummyVecEnv(
            [
                lambda: StockTradingEnv(
                    df=validation,
                    stock_dim=self.stock_dim,
                    hmax=self.hmax,
                    initial_amount=self.initial_amount,
                    num_stock_shares=[0] * self.stock_dim,
                    buy_cost_pct=[self.buy_cost_pct] * self.stock_dim,
                    sell_cost_pct=[self.sell_cost_pct] * self.stock_dim,
                    reward_scaling=self.reward_scaling,
                    state_space=self.state_space,
                    action_space=self.action_space,
                    tech_indicator_list=self.tech_indicator_list,
                    turbulence_threshold=turbulence_threshold,
                    iteration=i,
                    model_name=model_name,
                    mode="validation",
                    print_verbosity=self.print_verbosity,
                )
            ]
        )
        val_obs = val_env.reset()
        self.DRL_validation(
            model=model,
            test_data=validation,
            test_env=val_env,
            test_obs=val_obs,
        )
        sharpe = self.get_validation_sharpe(i, model_name=model_name)
        print(f"{model_name} Sharpe Ratio: ", sharpe)
        sharpe_list.append(sharpe)
        return model, sharpe_list, sharpe

    # def run_ensemble_strategy(self,
    #     A2C_model_kwargs=None,
    #     PPO_model_kwargs=None,
    #     DDPG_model_kwargs=None,
    #     SAC_model_kwargs=None,
    #     TD3_model_kwargs=None,
    #     Recurrent_model_kwargs=None,
    #     timesteps_dict=None,
    # ):
    #     # Model Parameters
    #     kwargs = {
    #         "a2c": A2C_model_kwargs,
    #         "ppo": PPO_model_kwargs,
    #         "ddpg": DDPG_model_kwargs,
    #         "sac": SAC_model_kwargs,
    #         "td3": TD3_model_kwargs,
    #         "re_ppo": Recurrent_model_kwargs,
    #     }
    #     # Model Sharpe Ratios
    #     model_dct = {k: {"sharpe_list": [], "sharpe": -1} for k in MODELS.keys()}

    #     """Ensemble Strategy that combines A2C, PPO, DDPG, SAC, and TD3"""
    #     print("============Start Ensemble Strategy============")
    #     # for ensemble model, it's necessary to feed the last state
    #     # of the previous model to the current model as the initial state
    #     last_state_ensemble = []

    #     model_use = []
    #     validation_start_date_list = []
    #     validation_end_date_list = []
    #     iteration_list = []

    #     insample_turbulence = self.df[
    #         (self.df.date < self.train_period[1])
    #         & (self.df.date >= self.train_period[0])
    #     ]
    #     insample_turbulence_threshold = np.quantile(
    #         insample_turbulence.turbulence.values, 0.90
    #     )

    #     start = time.time()
    #     for i in range(
    #         self.rebalance_window + self.validation_window,
    #         len(self.unique_trade_date),
    #         self.rebalance_window,
    #     ):
    #         validation_start_date = self.unique_trade_date[
    #             i - self.rebalance_window - self.validation_window
    #         ]
    #         validation_end_date = self.unique_trade_date[i - self.rebalance_window]

    #         validation_start_date_list.append(validation_start_date)
    #         validation_end_date_list.append(validation_end_date)
    #         iteration_list.append(i)

    #         print("============================================")
    #         # initial state is empty
    #         if i - self.rebalance_window - self.validation_window == 0:
    #             # inital state
    #             initial = True
    #         else:
    #             # previous state
    #             initial = False

    #         # Tuning trubulence index based on historical data
    #         # Turbulence lookback window is one quarter (63 days)
    #         end_date_index = self.df.index[
    #             self.df["date"]
    #             == self.unique_trade_date[
    #                 i - self.rebalance_window - self.validation_window
    #             ]
    #         ].to_list()[-1]
    #         start_date_index = end_date_index - 63 + 1

    #         historical_turbulence = self.df.iloc[
    #             start_date_index : (end_date_index + 1), :
    #         ]

    #         historical_turbulence = historical_turbulence.drop_duplicates(
    #             subset=["date"]
    #         )

    #         historical_turbulence_mean = np.mean(
    #             historical_turbulence.turbulence.values
    #         )

    #         # print(historical_turbulence_mean)

    #         if historical_turbulence_mean > insample_turbulence_threshold:
    #             # if the mean of the historical data is greater than the 90% quantile of insample turbulence data
    #             # then we assume that the current market is volatile,
    #             # therefore we set the 90% quantile of insample turbulence data as the turbulence threshold
    #             # meaning the current turbulence can't exceed the 90% quantile of insample turbulence data
    #             turbulence_threshold = insample_turbulence_threshold
    #         else:
    #             # if the mean of the historical data is less than the 90% quantile of insample turbulence data
    #             # then we tune up the turbulence_threshold, meaning we lower the risk
    #             turbulence_threshold = np.quantile(
    #                 insample_turbulence.turbulence.values, 1
    #             )

    #         turbulence_threshold = np.quantile(
    #             insample_turbulence.turbulence.values, 0.99
    #         )
    #         print("turbulence_threshold: ", turbulence_threshold)

    #         # Environment Setup starts
    #         # training env
    #         train = data_split(
    #             self.df,
    #             start=self.train_period[0],
    #             end=self.unique_trade_date[
    #                 i - self.rebalance_window - self.validation_window
    #             ],
    #         )
    #         self.train_env = DummyVecEnv(
    #             [
    #                 lambda: StockTradingEnv(
    #                     df=train,
    #                     stock_dim=self.stock_dim,
    #                     hmax=self.hmax,
    #                     initial_amount=self.initial_amount,
    #                     num_stock_shares=[0] * self.stock_dim,
    #                     buy_cost_pct=[self.buy_cost_pct] * self.stock_dim,
    #                     sell_cost_pct=[self.sell_cost_pct] * self.stock_dim,
    #                     reward_scaling=self.reward_scaling,
    #                     state_space=self.state_space,
    #                     action_space=self.action_space,
    #                     tech_indicator_list=self.tech_indicator_list,
    #                     print_verbosity=self.print_verbosity,
    #                 )
    #             ]
    #         )

    #         validation = data_split(
    #             self.df,
    #             start=self.unique_trade_date[
    #                 i - self.rebalance_window - self.validation_window
    #             ],
    #             end=self.unique_trade_date[i - self.rebalance_window],
    #         )
    #         # Environment Setup ends

    #         # Training and Validation starts
    #         print(
    #             "======Model training from: ",
    #             self.train_period[0],
    #             "to ",
    #             self.unique_trade_date[
    #                 i - self.rebalance_window - self.validation_window
    #             ],
    #         )
    #         # print("training: ",len(data_split(df, start=20090000, end=test.datadate.unique()[i-rebalance_window]) ))
    #         # print("==============Model Training===========")
    #         # Train Each Model
    #         trained_models = {k: v for k, v in kwargs.items() if v is not None}
    #         for model_name in trained_models.keys():
    #             model, sharpe_list, sharpe = self._train_window(
    #                 model_name,
    #                 trained_models[model_name],
    #                 model_dct[model_name]["sharpe_list"],
    #                 validation_start_date,
    #                 validation_end_date,
    #                 timesteps_dict,
    #                 i,
    #                 validation,
    #                 turbulence_threshold,
    #             )
    #             model_dct[model_name]["sharpe_list"] = sharpe_list
    #             model_dct[model_name]["model"] = model
    #             model_dct[model_name]["sharpe"] = sharpe

    #         print(
    #             "======Best Model Retraining from: ",
    #             self.train_period[0],
    #             "to ",
    #             self.unique_trade_date[i - self.rebalance_window],
    #         )
    #         # Environment setup for model retraining up to first trade date
    #         # train_full = data_split(self.df, start=self.train_period[0],
    #         # end=self.unique_trade_date[i - self.rebalance_window])
    #         # self.train_full_env = DummyVecEnv([lambda: StockTradingEnv(train_full,
    #         #                                               self.stock_dim,
    #         #                                               self.hmax,
    #         #                                               self.initial_amount,
    #         #                                               self.buy_cost_pct,
    #         #                                               self.sell_cost_pct,
    #         #                                               self.reward_scaling,
    #         #                                               self.state_space,
    #         #                                               self.action_space,
    #         #                                               self.tech_indicator_list,
    #         #                                              print_verbosity=self.print_verbosity
    #         # )])
    #         # Model Selection based on sharpe ratio
    #         # Same order as MODELS: {"a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO}
    #         sharpes = [model_dct[k]["sharpe"] for k in MODELS.keys()]
    #         # Find the model with the highest sharpe ratio
    #         max_mod = list(MODELS.keys())[np.argmax(sharpes)]
    #         model_use.append(max_mod.upper())
    #         model_ensemble = model_dct[max_mod]["model"]
    #         # Training and Validation ends

    #         # Trading starts
    #         print(
    #             "======Trading from: ",
    #             self.unique_trade_date[i - self.rebalance_window],
    #             "to ",
    #             self.unique_trade_date[i],
    #         )
    #         # print("Used Model: ", model_ensemble)
    #         last_state_ensemble = self.DRL_prediction(
    #             model=model_ensemble,
    #             name="ensemble",
    #             last_state=last_state_ensemble,
    #             iter_num=i,
    #             turbulence_threshold=turbulence_threshold,
    #             initial=initial,
    #         )
    #         # Trading ends

    #     end = time.time()
    #     print("Ensemble Strategy took: ", (end - start) / 60, " minutes")

    #     df_summary = pd.DataFrame(
    #         [
    #             iteration_list,
    #             validation_start_date_list,
    #             validation_end_date_list,
    #             model_use,
    #             model_dct["a2c"]["sharpe_list"],
    #             model_dct["ppo"]["sharpe_list"],
    #             model_dct["ddpg"]["sharpe_list"],
    #             model_dct["sac"]["sharpe_list"],
    #             model_dct["td3"]["sharpe_list"],
    #             model_dct["re_ppo"]["sharpe_list"],
    #         ]
    #     ).T
    #     df_summary.columns = [
    #         "Iter",
    #         "Val Start",
    #         "Val End",
    #         "Model Used",
    #         "A2C Sharpe",
    #         "PPO Sharpe",
    #         "DDPG Sharpe",
    #         "SAC Sharpe",
    #         "TD3 Sharpe",
    #         "RecurrentPPO Sharpe",
    #     ]

    #     return df_summary
    
    def run_ensemble_strategy(
        self,
        A2C_model_kwargs=None,
        PPO_model_kwargs=None,
        DDPG_model_kwargs=None,
        SAC_model_kwargs=None,
        TD3_model_kwargs=None,
        Recurrent_model_kwargs=None,
        timesteps_dict=None,
    ):
        # Model Parameters
        kwargs = {
            "a2c": A2C_model_kwargs,
            "ppo": PPO_model_kwargs,
            "ddpg": DDPG_model_kwargs,
            "sac": SAC_model_kwargs,
            "td3": TD3_model_kwargs,
            "re_ppo": Recurrent_model_kwargs,
        }
        # Chỉ tạo model_dct cho các mô hình có model_kwargs
        trained_models = {k: v for k, v in kwargs.items() if v is not None}
        model_dct = {k: {"sharpe_list": [], "sharpe": -1} for k in trained_models.keys()}

        if not trained_models:
            raise ValueError("No model kwargs provided for training.")

        print("============Start Ensemble Strategy============")
        # for ensemble model, it's necessary to feed the last state
        # of the previous model to the current model as the initial state
        last_state_ensemble = []

        model_use = []
        validation_start_date_list = []
        validation_end_date_list = []
        iteration_list = []

        insample_turbulence = self.df[
            (self.df.date < self.train_period[1])
            & (self.df.date >= self.train_period[0])
        ]
        insample_turbulence_threshold = np.quantile(
            insample_turbulence.turbulence.values, 0.90
        )

        start = time.time()
        for i in range(
            self.rebalance_window + self.validation_window,
            len(self.unique_trade_date),
            self.rebalance_window,
        ):
            validation_start_date = self.unique_trade_date[
                i - self.rebalance_window - self.validation_window
            ]
            validation_end_date = self.unique_trade_date[i - self.rebalance_window]

            validation_start_date_list.append(validation_start_date)
            validation_end_date_list.append(validation_end_date)
            iteration_list.append(i)

            print("============================================")
            # initial state is empty
            if i - self.rebalance_window - self.validation_window == 0:
                initial = True
            else:
                initial = False

            # Tuning turbulence index based on historical data
            end_date_index = self.df.index[
                self.df["date"]
                == self.unique_trade_date[i - self.rebalance_window - self.validation_window]
            ].to_list()[-1]
            start_date_index = end_date_index - 63 + 1

            historical_turbulence = self.df.iloc[start_date_index:(end_date_index + 1), :]

            historical_turbulence = historical_turbulence.drop_duplicates(subset=["date"])

            historical_turbulence_mean = np.mean(historical_turbulence.turbulence.values)

            if historical_turbulence_mean > insample_turbulence_threshold:
                turbulence_threshold = insample_turbulence_threshold
            else:
                turbulence_threshold = np.quantile(
                    insample_turbulence.turbulence.values, 1
                )

            turbulence_threshold = np.quantile(
                insample_turbulence.turbulence.values, 0.99
            )
            print("turbulence_threshold: ", turbulence_threshold)

            # Environment Setup
            train = data_split(
                self.df,
                start=self.train_period[0],
                end=self.unique_trade_date[i - self.rebalance_window - self.validation_window],
            )
            self.train_env = DummyVecEnv(
                [
                    lambda: StockTradingEnv(
                        df=train,
                        stock_dim=self.stock_dim,
                        hmax=self.hmax,
                        initial_amount=self.initial_amount,
                        num_stock_shares=[0] * self.stock_dim,
                        buy_cost_pct=[self.buy_cost_pct] * self.stock_dim,
                        sell_cost_pct=[self.sell_cost_pct] * self.stock_dim,
                        reward_scaling=self.reward_scaling,
                        state_space=self.state_space,
                        action_space=self.action_space,
                        tech_indicator_list=self.tech_indicator_list,
                        print_verbosity=self.print_verbosity,
                    )
                ]
            )

            validation = data_split(
                self.df,
                start=self.unique_trade_date[i - self.rebalance_window - self.validation_window],
                end=self.unique_trade_date[i - self.rebalance_window],
            )

            # Training and Validation
            print(
                "======Model training from: ",
                self.train_period[0],
                "to ",
                self.unique_trade_date[i - self.rebalance_window - self.validation_window],
            )
            for model_name in trained_models.keys():
                model, sharpe_list, sharpe = self._train_window(
                    model_name,
                    trained_models[model_name],
                    model_dct[model_name]["sharpe_list"],
                    validation_start_date,
                    validation_end_date,
                    timesteps_dict,
                    i,
                    validation,
                    turbulence_threshold,
                )
                model_dct[model_name]["sharpe_list"] = sharpe_list
                model_dct[model_name]["model"] = model
                model_dct[model_name]["sharpe"] = sharpe

            print(
                "======Best Model Retraining from: ",
                self.train_period[0],
                "to ",
                self.unique_trade_date[i - self.rebalance_window],
            )
            sharpes = [model_dct[k]["sharpe"] for k in trained_models.keys()]
            max_mod = list(trained_models.keys())[np.argmax(sharpes)]
            model_use.append(max_mod.upper())
            model_ensemble = model_dct[max_mod]["model"]

            # Trading
            print(
                "======Trading from: ",
                self.unique_trade_date[i - self.rebalance_window],
                "to ",
                self.unique_trade_date[i],
            )
            last_state_ensemble = self.DRL_prediction(
                model=model_ensemble,
                name="ensemble",
                last_state=last_state_ensemble,
                iter_num=i,
                turbulence_threshold=turbulence_threshold,
                initial=initial,
            )

        end = time.time()
        print("Ensemble Strategy took: ", (end - start) / 60, " minutes")

        # Tạo cột cho các mô hình đã huấn luyện
        columns = ["Iter", "Val Start", "Val End", "Model Used"]
        sharpe_columns = [f"{k.upper()} Sharpe" for k in trained_models.keys()]
        df_summary = pd.DataFrame(
            [iteration_list, validation_start_date_list, validation_end_date_list, model_use]
            + [model_dct[k]["sharpe_list"] for k in trained_models.keys()]
        ).T
        df_summary.columns = columns + sharpe_columns

        return df_summary