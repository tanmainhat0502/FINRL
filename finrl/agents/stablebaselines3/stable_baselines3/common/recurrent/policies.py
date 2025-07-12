# from typing import Any, Optional, Union

# import numpy as np
# import torch as th
# from gymnasium import spaces
# from stable_baselines3.common.distributions import Distribution
# from stable_baselines3.common.policies import ActorCriticPolicy
# from stable_baselines3.common.torch_layers import (
#     BaseFeaturesExtractor,
#     CombinedExtractor,
#     FlattenExtractor,
#     MlpExtractor,
#     NatureCNN,
# )
# from xlstm import (
#     xLSTMBlockStack,
#     xLSTMBlockStackConfig,
#     mLSTMBlockConfig,
#     mLSTMLayerConfig,
#     sLSTMBlockConfig,
#     sLSTMLayerConfig,
#     FeedForwardConfig,
# )


# from stable_baselines3.common.type_aliases import Schedule
# from stable_baselines3.common.utils import zip_strict
# from torch import nn
# from finrl.agents.stablebaselines3.stable_baselines3.common.recurrent.type_aliases import RNNStates



# class RecurrentActorCriticPolicy(ActorCriticPolicy):
#     """
#     Recurrent policy class for actor-critic algorithms (has both policy and value prediction).
#     To be used with A2C, PPO and the likes.
#     It assumes that both the actor and the critic LSTM
#     have the same architecture.

#     :param observation_space: Observation space
#     :param action_space: Action space
#     :param lr_schedule: Learning rate schedule (could be constant)
#     :param net_arch: The specification of the policy and value networks.
#     :param activation_fn: Activation function
#     :param ortho_init: Whether to use or not orthogonal initialization
#     :param use_sde: Whether to use State Dependent Exploration or not
#     :param log_std_init: Initial value for the log standard deviation
#     :param full_std: Whether to use (n_features x n_actions) parameters
#         for the std instead of only (n_features,) when using gSDE
#     :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
#         a positive standard deviation (cf paper). It allows to keep variance
#         above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
#     :param squash_output: Whether to squash the output using a tanh function,
#         this allows to ensure boundaries when using gSDE.
#     :param features_extractor_class: Features extractor to use.
#     :param features_extractor_kwargs: Keyword arguments
#         to pass to the features extractor.
#     :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
#     :param normalize_images: Whether to normalize images or not,
#          dividing by 255.0 (True by default)
#     :param optimizer_class: The optimizer to use,
#         ``th.optim.Adam`` by default
#     :param optimizer_kwargs: Additional keyword arguments,
#         excluding the learning rate, to pass to the optimizer
#     :param lstm_hidden_size: Number of hidden units for each LSTM layer.
#     :param n_lstm_layers: Number of LSTM layers.
#     :param shared_lstm: Whether the LSTM is shared between the actor and the critic
#         (in that case, only the actor gradient is used)
#         By default, the actor and the critic have two separate LSTM.
#     :param enable_critic_lstm: Use a seperate LSTM for the critic.
#     :param lstm_kwargs: Additional keyword arguments to pass the the LSTM
#         constructor.
#     """

#     def __init__(
#         self,
#         observation_space: spaces.Space,
#         action_space: spaces.Space,
#         lr_schedule: Schedule,
#         net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
#         activation_fn: type[nn.Module] = nn.Tanh,
#         ortho_init: bool = True,
#         use_sde: bool = False,
#         log_std_init: float = 0.0,
#         full_std: bool = True,
#         use_expln: bool = False,
#         squash_output: bool = False,
#         features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
#         features_extractor_kwargs: Optional[dict[str, Any]] = None,
#         share_features_extractor: bool = True,
#         normalize_images: bool = True,
#         optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
#         optimizer_kwargs: Optional[dict[str, Any]] = None,
#         lstm_hidden_size: int = 256,
#         n_lstm_layers: int = 1,
#         shared_lstm: bool = False,
#         enable_critic_lstm: bool = True,
#         lstm_kwargs: Optional[dict[str, Any]] = None,
#     ):
#         self.lstm_output_dim = lstm_hidden_size
#         super().__init__(
#             observation_space,
#             action_space,
#             lr_schedule,
#             net_arch,
#             activation_fn,
#             ortho_init,
#             use_sde,
#             log_std_init,
#             full_std,
#             use_expln,
#             squash_output,
#             features_extractor_class,
#             features_extractor_kwargs,
#             share_features_extractor,
#             normalize_images,
#             optimizer_class,
#             optimizer_kwargs,
#         )

#         self.lstm_kwargs = lstm_kwargs or {}
#         self.shared_lstm = shared_lstm
#         self.enable_critic_lstm = enable_critic_lstm
    
        
#         self.lstm_actor = nn.LSTM(
#             self.features_dim,
#             lstm_hidden_size,
#             num_layers=n_lstm_layers,
#             **self.lstm_kwargs,
#         )
#         # For the predict() method, to initialize hidden states
#         # (n_lstm_layers, batch_size, lstm_hidden_size)

#         self.lstm_hidden_state_shape = (n_lstm_layers, 1, lstm_hidden_size)
#         self.critic = None
#         self.lstm_critic = None
#         assert not (
#             self.shared_lstm and self.enable_critic_lstm
#         ), "You must choose between shared LSTM, seperate or no LSTM for the critic."

#         assert not (
#             self.shared_lstm and not self.share_features_extractor
#         ), "If the features extractor is not shared, the LSTM cannot be shared."

#         # No LSTM for the critic, we still need to convert
#         # output of features extractor to the correct size
#         # (size of the output of the actor lstm)
#         if not (self.shared_lstm or self.enable_critic_lstm):
#             self.critic = nn.Linear(self.features_dim, lstm_hidden_size)

#         # Use a separate LSTM for the critic
#         if self.enable_critic_lstm:
#             self.lstm_critic = nn.LSTM(
#                 self.features_dim,
#                 lstm_hidden_size,
#                 num_layers=n_lstm_layers,
#                 **self.lstm_kwargs,
#             )

#         # Setup optimizer with initial learning rate
#         self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

#     def _build_mlp_extractor(self) -> None:
#         """
#         Create the policy and value networks.
#         Part of the layers can be shared.
#         """
#         self.mlp_extractor = MlpExtractor(
#             self.lstm_output_dim,
#             net_arch=self.net_arch,
#             activation_fn=self.activation_fn,
#             device=self.device,
#         )

#     @staticmethod
#     def _process_sequence(
#         features: th.Tensor,
#         lstm_states: tuple[th.Tensor, th.Tensor],
#         episode_starts: th.Tensor,
#         lstm: nn.LSTM,
#     ) -> tuple[th.Tensor, th.Tensor]:
#         """
#         Do a forward pass in the LSTM network.

#         :param features: Input tensor
#         :param lstm_states: previous hidden and cell states of the LSTM, respectively
#         :param episode_starts: Indicates when a new episode starts,
#             in that case, we need to reset LSTM states.
#         :param lstm: LSTM object.
#         :return: LSTM output and updated LSTM states.
#         """
#         # LSTM logic
#         # (sequence length, batch size, features dim)
#         # (batch size = n_envs for data collection or n_seq when doing gradient update)
#         n_seq = lstm_states[0].shape[1]
#         # Batch to sequence
#         # (padded batch size, features_dim) -> (n_seq, max length, features_dim) -> (max length, n_seq, features_dim)
#         # note: max length (max sequence length) is always 1 during data collection
#         features_sequence = features.reshape((n_seq, -1, lstm.input_size)).swapaxes(0, 1)
#         episode_starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)

#         # If we don't have to reset the state in the middle of a sequence
#         # we can avoid the for loop, which speeds up things
#         if th.all(episode_starts == 0.0):
#             lstm_output, lstm_states = lstm(features_sequence, lstm_states)
#             lstm_output = th.flatten(lstm_output.transpose(0, 1), start_dim=0, end_dim=1)
#             return lstm_output, lstm_states

#         lstm_output = []
#         # Iterate over the sequence
#         for features, episode_start in zip_strict(features_sequence, episode_starts):
#             hidden, lstm_states = lstm(
#                 features.unsqueeze(dim=0),
#                 (
#                     # Reset the states at the beginning of a new episode
#                     (1.0 - episode_start).view(1, n_seq, 1) * lstm_states[0],
#                     (1.0 - episode_start).view(1, n_seq, 1) * lstm_states[1],
#                 ),
#             )
#             lstm_output += [hidden]
#         # Sequence to batch
#         # (sequence length, n_seq, lstm_out_dim) -> (batch_size, lstm_out_dim)
#         lstm_output = th.flatten(th.cat(lstm_output).transpose(0, 1), start_dim=0, end_dim=1)
#         return lstm_output, lstm_states

#     def forward(
#         self,
#         obs: th.Tensor,
#         lstm_states: RNNStates,
#         episode_starts: th.Tensor,
#         deterministic: bool = False,
#     ) -> tuple[th.Tensor, th.Tensor, th.Tensor, RNNStates]:
#         """
#         Forward pass in all the networks (actor and critic)

#         :param obs: Observation. Observation
#         :param lstm_states: The last hidden and memory states for the LSTM.
#         :param episode_starts: Whether the observations correspond to new episodes
#             or not (we reset the lstm states in that case).
#         :param deterministic: Whether to sample or use deterministic actions
#         :return: action, value and log probability of the action
#         """
#         # Preprocess the observation if needed
#         features = self.extract_features(obs)
#         if self.share_features_extractor:
#             pi_features = vf_features = features  # alis
#         else:
#             pi_features, vf_features = features
#         # latent_pi, latent_vf = self.mlp_extractor(features)
#         latent_pi, lstm_states_pi = self._process_sequence(pi_features, lstm_states.pi, episode_starts, self.lstm_actor)
#         if self.lstm_critic is not None:
#             latent_vf, lstm_states_vf = self._process_sequence(vf_features, lstm_states.vf, episode_starts, self.lstm_critic)
#         elif self.shared_lstm:
#             # Re-use LSTM features but do not backpropagate
#             latent_vf = latent_pi.detach()
#             lstm_states_vf = (lstm_states_pi[0].detach(), lstm_states_pi[1].detach())
#         else:
#             # Critic only has a feedforward network
#             latent_vf = self.critic(vf_features)
#             lstm_states_vf = lstm_states_pi

#         latent_pi = self.mlp_extractor.forward_actor(latent_pi)
#         latent_vf = self.mlp_extractor.forward_critic(latent_vf)

#         # Evaluate the values for the given observations
#         values = self.value_net(latent_vf)
#         distribution = self._get_action_dist_from_latent(latent_pi)
#         actions = distribution.get_actions(deterministic=deterministic)
#         log_prob = distribution.log_prob(actions)
#         return actions, values, log_prob, RNNStates(lstm_states_pi, lstm_states_vf)

#     def get_distribution(
#         self,
#         obs: th.Tensor,
#         lstm_states: tuple[th.Tensor, th.Tensor],
#         episode_starts: th.Tensor,
#     ) -> tuple[Distribution, tuple[th.Tensor, ...]]:
#         """
#         Get the current policy distribution given the observations.

#         :param obs: Observation.
#         :param lstm_states: The last hidden and memory states for the LSTM.
#         :param episode_starts: Whether the observations correspond to new episodes
#             or not (we reset the lstm states in that case).
#         :return: the action distribution and new hidden states.
#         """
#         # Call the method from the parent of the parent class
#         features = super(ActorCriticPolicy, self).extract_features(obs, self.pi_features_extractor)
#         latent_pi, lstm_states = self._process_sequence(features, lstm_states, episode_starts, self.lstm_actor)
#         latent_pi = self.mlp_extractor.forward_actor(latent_pi)
#         return self._get_action_dist_from_latent(latent_pi), lstm_states

#     def predict_values(
#         self,
#         obs: th.Tensor,
#         lstm_states: tuple[th.Tensor, th.Tensor],
#         episode_starts: th.Tensor,
#     ) -> th.Tensor:
#         """
#         Get the estimated values according to the current policy given the observations.

#         :param obs: Observation.
#         :param lstm_states: The last hidden and memory states for the LSTM.
#         :param episode_starts: Whether the observations correspond to new episodes
#             or not (we reset the lstm states in that case).
#         :return: the estimated values.
#         """
#         # Call the method from the parent of the parent class
#         features = super(ActorCriticPolicy, self).extract_features(obs, self.vf_features_extractor)

#         if self.lstm_critic is not None:
#             latent_vf, lstm_states_vf = self._process_sequence(features, lstm_states, episode_starts, self.lstm_critic)
#         elif self.shared_lstm:
#             # Use LSTM from the actor
#             latent_pi, _ = self._process_sequence(features, lstm_states, episode_starts, self.lstm_actor)
#             latent_vf = latent_pi.detach()
#         else:
#             latent_vf = self.critic(features)

#         latent_vf = self.mlp_extractor.forward_critic(latent_vf)
#         return self.value_net(latent_vf)

#     def evaluate_actions(
#         self, obs: th.Tensor, actions: th.Tensor, lstm_states: RNNStates, episode_starts: th.Tensor
#     ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
#         """
#         Evaluate actions according to the current policy,
#         given the observations.

#         :param obs: Observation.
#         :param actions:
#         :param lstm_states: The last hidden and memory states for the LSTM.
#         :param episode_starts: Whether the observations correspond to new episodes
#             or not (we reset the lstm states in that case).
#         :return: estimated value, log likelihood of taking those actions
#             and entropy of the action distribution.
#         """
#         # Preprocess the observation if needed
#         features = self.extract_features(obs)
#         if self.share_features_extractor:
#             pi_features = vf_features = features  # alias
#         else:
#             pi_features, vf_features = features
#         latent_pi, _ = self._process_sequence(pi_features, lstm_states.pi, episode_starts, self.lstm_actor)
#         if self.lstm_critic is not None:
#             latent_vf, _ = self._process_sequence(vf_features, lstm_states.vf, episode_starts, self.lstm_critic)
#         elif self.shared_lstm:
#             latent_vf = latent_pi.detach()
#         else:
#             latent_vf = self.critic(vf_features)

#         latent_pi = self.mlp_extractor.forward_actor(latent_pi)
#         latent_vf = self.mlp_extractor.forward_critic(latent_vf)

#         distribution = self._get_action_dist_from_latent(latent_pi)
#         log_prob = distribution.log_prob(actions)
#         values = self.value_net(latent_vf)
#         return values, log_prob, distribution.entropy()

#     def _predict(
#         self,
#         observation: th.Tensor,
#         lstm_states: tuple[th.Tensor, th.Tensor],
#         episode_starts: th.Tensor,
#         deterministic: bool = False,
#     ) -> tuple[th.Tensor, tuple[th.Tensor, ...]]:
#         """
#         Get the action according to the policy for a given observation.

#         :param observation:
#         :param lstm_states: The last hidden and memory states for the LSTM.
#         :param episode_starts: Whether the observations correspond to new episodes
#             or not (we reset the lstm states in that case).
#         :param deterministic: Whether to use stochastic or deterministic actions
#         :return: Taken action according to the policy and hidden states of the RNN
#         """
#         distribution, lstm_states = self.get_distribution(observation, lstm_states, episode_starts)
#         return distribution.get_actions(deterministic=deterministic), lstm_states

#     def predict(
#         self,
#         observation: Union[np.ndarray, dict[str, np.ndarray]],
#         state: Optional[tuple[np.ndarray, ...]] = None,
#         episode_start: Optional[np.ndarray] = None,
#         deterministic: bool = False,
#     ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
#         """
#         Get the policy action from an observation (and optional hidden state).
#         Includes sugar-coating to handle different observations (e.g. normalizing images).

#         :param observation: the input observation
#         :param lstm_states: The last hidden and memory states for the LSTM.
#         :param episode_starts: Whether the observations correspond to new episodes
#             or not (we reset the lstm states in that case).
#         :param deterministic: Whether or not to return deterministic actions.
#         :return: the model's action and the next hidden state
#             (used in recurrent policies)
#         """
#         # Switch to eval mode (this affects batch norm / dropout)
#         self.set_training_mode(False)

#         observation, vectorized_env = self.obs_to_tensor(observation)

#         if isinstance(observation, dict):
#             n_envs = observation[next(iter(observation.keys()))].shape[0]
#         else:
#             n_envs = observation.shape[0]
#         # state : (n_layers, n_envs, dim)
#         if state is None:
#             # Initialize hidden states to zeros
#             state = np.concatenate([np.zeros(self.lstm_hidden_state_shape) for _ in range(n_envs)], axis=1)
#             state = (state, state)

#         if episode_start is None:
#             episode_start = np.array([False for _ in range(n_envs)])

#         with th.no_grad():
#             # Convert to PyTorch tensors
#             states = th.tensor(state[0], dtype=th.float32, device=self.device), th.tensor(
#                 state[1], dtype=th.float32, device=self.device
#             )
#             episode_starts = th.tensor(episode_start, dtype=th.float32, device=self.device)
#             actions, states = self._predict(
#                 observation, lstm_states=states, episode_starts=episode_starts, deterministic=deterministic
#             )
#             states = (states[0].cpu().numpy(), states[1].cpu().numpy())

#         # Convert to numpy
#         actions = actions.cpu().numpy()

#         if isinstance(self.action_space, spaces.Box):
#             if self.squash_output:
#                 # Rescale to proper domain when using squashing
#                 actions = self.unscale_action(actions)
#             else:
#                 # Actions could be on arbitrary scale, so clip the actions to avoid
#                 # out of bound error (e.g. if sampling from a Gaussian distribution)
#                 actions = np.clip(actions, self.action_space.low, self.action_space.high)

#         # Remove batch dimension if needed
#         if not vectorized_env:
#             actions = actions.squeeze(axis=0)

#         return actions, states


# class RecurrentActorCriticCnnPolicy(RecurrentActorCriticPolicy):
#     """
#     CNN recurrent policy class for actor-critic algorithms (has both policy and value prediction).
#     Used by A2C, PPO and the likes.

#     :param observation_space: Observation space
#     :param action_space: Action space
#     :param lr_schedule: Learning rate schedule (could be constant)
#     :param net_arch: The specification of the policy and value networks.
#     :param activation_fn: Activation function
#     :param ortho_init: Whether to use or not orthogonal initialization
#     :param use_sde: Whether to use State Dependent Exploration or not
#     :param log_std_init: Initial value for the log standard deviation
#     :param full_std: Whether to use (n_features x n_actions) parameters
#         for the std instead of only (n_features,) when using gSDE
#     :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
#         a positive standard deviation (cf paper). It allows to keep variance
#         above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
#     :param squash_output: Whether to squash the output using a tanh function,
#         this allows to ensure boundaries when using gSDE.
#     :param features_extractor_class: Features extractor to use.
#     :param features_extractor_kwargs: Keyword arguments
#         to pass to the features extractor.
#     :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
#     :param normalize_images: Whether to normalize images or not,
#          dividing by 255.0 (True by default)
#     :param optimizer_class: The optimizer to use,
#         ``th.optim.Adam`` by default
#     :param optimizer_kwargs: Additional keyword arguments,
#         excluding the learning rate, to pass to the optimizer
#     :param lstm_hidden_size: Number of hidden units for each LSTM layer.
#     :param n_lstm_layers: Number of LSTM layers.
#     :param shared_lstm: Whether the LSTM is shared between the actor and the critic.
#         By default, only the actor has a recurrent network.
#     :param enable_critic_lstm: Use a seperate LSTM for the critic.
#     :param lstm_kwargs: Additional keyword arguments to pass the the LSTM
#         constructor.
#     """

#     def __init__(
#         self,
#         observation_space: spaces.Space,
#         action_space: spaces.Space,
#         lr_schedule: Schedule,
#         net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
#         activation_fn: type[nn.Module] = nn.Tanh,
#         ortho_init: bool = True,
#         use_sde: bool = False,
#         log_std_init: float = 0.0,
#         full_std: bool = True,
#         use_expln: bool = False,
#         squash_output: bool = False,
#         features_extractor_class: type[BaseFeaturesExtractor] = NatureCNN,
#         features_extractor_kwargs: Optional[dict[str, Any]] = None,
#         share_features_extractor: bool = True,
#         normalize_images: bool = True,
#         optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
#         optimizer_kwargs: Optional[dict[str, Any]] = None,
#         lstm_hidden_size: int = 256,
#         n_lstm_layers: int = 1,
#         shared_lstm: bool = False,
#         enable_critic_lstm: bool = True,
#         lstm_kwargs: Optional[dict[str, Any]] = None,
#     ):
#         super().__init__(
#             observation_space,
#             action_space,
#             lr_schedule,
#             net_arch,
#             activation_fn,
#             ortho_init,
#             use_sde,
#             log_std_init,
#             full_std,
#             use_expln,
#             squash_output,
#             features_extractor_class,
#             features_extractor_kwargs,
#             share_features_extractor,
#             normalize_images,
#             optimizer_class,
#             optimizer_kwargs,
#             lstm_hidden_size,
#             n_lstm_layers,
#             shared_lstm,
#             enable_critic_lstm,
#             lstm_kwargs,
#         )


# class RecurrentMultiInputActorCriticPolicy(RecurrentActorCriticPolicy):
#     """
#     MultiInputActorClass policy class for actor-critic algorithms (has both policy and value prediction).
#     Used by A2C, PPO and the likes.

#     :param observation_space: Observation space
#     :param action_space: Action space
#     :param lr_schedule: Learning rate schedule (could be constant)
#     :param net_arch: The specification of the policy and value networks.
#     :param activation_fn: Activation function
#     :param ortho_init: Whether to use or not orthogonal initialization
#     :param use_sde: Whether to use State Dependent Exploration or not
#     :param log_std_init: Initial value for the log standard deviation
#     :param full_std: Whether to use (n_features x n_actions) parameters
#         for the std instead of only (n_features,) when using gSDE
#     :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
#         a positive standard deviation (cf paper). It allows to keep variance
#         above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
#     :param squash_output: Whether to squash the output using a tanh function,
#         this allows to ensure boundaries when using gSDE.
#     :param features_extractor_class: Features extractor to use.
#     :param features_extractor_kwargs: Keyword arguments
#         to pass to the features extractor.
#     :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
#     :param normalize_images: Whether to normalize images or not,
#          dividing by 255.0 (True by default)
#     :param optimizer_class: The optimizer to use,
#         ``th.optim.Adam`` by default
#     :param optimizer_kwargs: Additional keyword arguments,
#         excluding the learning rate, to pass to the optimizer
#     :param lstm_hidden_size: Number of hidden units for each LSTM layer.
#     :param n_lstm_layers: Number of LSTM layers.
#     :param shared_lstm: Whether the LSTM is shared between the actor and the critic.
#         By default, only the actor has a recurrent network.
#     :param enable_critic_lstm: Use a seperate LSTM for the critic.
#     :param lstm_kwargs: Additional keyword arguments to pass the the LSTM
#         constructor.
#     """

#     def __init__(
#         self,
#         observation_space: spaces.Space,
#         action_space: spaces.Space,
#         lr_schedule: Schedule,
#         net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
#         activation_fn: type[nn.Module] = nn.Tanh,
#         ortho_init: bool = True,
#         use_sde: bool = False,
#         log_std_init: float = 0.0,
#         full_std: bool = True,
#         use_expln: bool = False,
#         squash_output: bool = False,
#         features_extractor_class: type[BaseFeaturesExtractor] = CombinedExtractor,
#         features_extractor_kwargs: Optional[dict[str, Any]] = None,
#         share_features_extractor: bool = True,
#         normalize_images: bool = True,
#         optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
#         optimizer_kwargs: Optional[dict[str, Any]] = None,
#         lstm_hidden_size: int = 256,
#         n_lstm_layers: int = 1,
#         shared_lstm: bool = False,
#         enable_critic_lstm: bool = True,
#         lstm_kwargs: Optional[dict[str, Any]] = None,
#     ):
#         super().__init__(
#             observation_space,
#             action_space,
#             lr_schedule,
#             net_arch,
#             activation_fn,
#             ortho_init,
#             use_sde,
#             log_std_init,
#             full_std,
#             use_expln,
#             squash_output,
#             features_extractor_class,
#             features_extractor_kwargs,
#             share_features_extractor,
#             normalize_images,
#             optimizer_class,
#             optimizer_kwargs,
#             lstm_hidden_size,
#             n_lstm_layers,
#             shared_lstm,
#             enable_critic_lstm,
#             lstm_kwargs,
#         )






from typing import Any, Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
)

from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import zip_strict
from torch import nn
from finrl.agents.stablebaselines3.stable_baselines3.common.recurrent.type_aliases import RNNStates


from finrl.agents.stablebaselines3.stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn
import torch as th

from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)

from typing import Optional, Tuple, Union
class RecurrentActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 1,
        shared_lstm: bool = False,
        enable_critic_lstm: bool = True,
        lstm_kwargs: Optional[dict[str, Any]] = None,
        context_length: int = 61,  # Dựa trên log
    ):
        self.lstm_output_dim = lstm_hidden_size
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

        self.lstm_kwargs = lstm_kwargs or {}
        self.shared_lstm = shared_lstm
        self.enable_critic_lstm = enable_critic_lstm
        self.context_length = context_length

        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="cuda",
                    num_heads=4,
                    conv1d_kernel_size=4,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=context_length,
            num_blocks=n_lstm_layers,
            embedding_dim=lstm_hidden_size,
            slstm_at=[1],

        )

        self.xlstm_actor = xLSTMBlockStack(cfg).to("cuda")

        self.lstm_hidden_state_shape = (1, 1, lstm_hidden_size)
        self.critic = None
        self.xlstm_critic = None

        if self.enable_critic_lstm:
            self.xlstm_critic = xLSTMBlockStack(cfg).to("cuda")

        # Tính action_dim từ action_space
        action_dim = action_space.shape[0] if len(action_space.shape) > 0 else action_space.n
        self.action_net = nn.Linear(lstm_hidden_size, action_dim)  # Chỉ sử dụng action_dim
        self.log_std = nn.Parameter(th.ones(1, action_dim) * log_std_init)

        # Khởi tạo value_net với kích thước đầu vào khớp với lstm_hidden_size
        self.value_net = nn.Linear(lstm_hidden_size, 1)

        assert not (
            self.shared_lstm and self.enable_critic_lstm
        ), "You must choose between shared xLSTM, separate or no xLSTM for the critic."

        assert not (
            self.shared_lstm and not self.share_features_extractor
        ), "If the features extractor is not shared, the xLSTM cannot be shared."

        if not (self.shared_lstm or self.enable_critic_lstm):
            self.critic = nn.Linear(self.features_dim, lstm_hidden_size)

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = MlpExtractor(
            self.lstm_output_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def extract_features(self, obs: th.Tensor, lstm_states: Tuple[th.Tensor, ...] = None, episode_starts: th.Tensor = None) -> th.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0).unsqueeze(-1)
        elif obs.dim() == 2:
            obs = obs.unsqueeze(-1)
        elif obs.dim() == 3 and obs.shape[2] == 1:
            pass
        else:
            raise ValueError(f"Unexpected obs shape: {obs.shape}")

        if obs.shape[2] != self.lstm_output_dim:
            obs = obs.expand(-1, -1, self.lstm_output_dim)

        if obs.shape[1] != self.context_length:
            if obs.shape[1] < self.context_length:
                pad_length = self.context_length - obs.shape[1]
                padding = th.zeros((obs.shape[0], pad_length, obs.shape[2]), device=obs.device)
                obs = th.cat([obs, padding], dim=1)
            else:
                obs = obs[:, :self.context_length, :]

        print(f"Extracted features shape: {obs.shape}")
        return obs

    @staticmethod
    def _process_sequence(
        features: th.Tensor,
        lstm_states: Tuple[th.Tensor, ...],
        episode_starts: th.Tensor,
        xlstm: xLSTMBlockStack,
        context_length: int,
    ) -> Tuple[th.Tensor, Tuple[th.Tensor, ...]]:
        if len(features.shape) == 2:
            batch_size, features_dim = features.shape
            features = features.unsqueeze(1)
        else:
            batch_size, seq_length, features_dim = features.shape

        if seq_length < context_length:
            pad_length = context_length - seq_length
            padding = th.zeros((batch_size, pad_length, features_dim), device=features.device)
            features_sequence = th.cat([features, padding], dim=1)
        else:
            features_sequence = features[:, :context_length, :]

        xlstm_output = xlstm(features_sequence.to("cuda"))
        # Tạo dummy_states bằng cách lặp qua các tensor con trong lstm_states
        dummy_states = ()
        for state_tuple in lstm_states:
            if isinstance(state_tuple, tuple):
                dummy_states += tuple(th.zeros_like(state) for state in state_tuple)
            else:
                dummy_states += (th.zeros_like(state_tuple),)
        return xlstm_output, dummy_states

    def forward(self, obs, lstm_states: Tuple[th.Tensor, ...], episode_starts: th.Tensor, deterministic: bool = False):
        print(f"Input obs shape: {obs.shape}")
        features = self.extract_features(obs, lstm_states, episode_starts)
        if not isinstance(features, tuple):
            features = (features, features)
        pi_features, vf_features = features
        print(f"Extracted features shape: {pi_features.shape}")

        # Sử dụng lstm_states trực tiếp
        latent_pi, lstm_states_pi = self._process_sequence(pi_features, lstm_states, episode_starts, self.xlstm_actor, self.context_length)
        if self.xlstm_critic is not None and not self.shared_lstm:
            latent_vf, lstm_states_vf = self._process_sequence(vf_features, lstm_states, episode_starts, self.xlstm_critic, self.context_length)
        else:
            latent_vf, lstm_states_vf = latent_pi, lstm_states_pi

        # Giảm chiều latent_pi và latent_vf
        latent_pi = latent_pi.mean(dim=1)  # Giảm từ [1, 61, 256] thành [1, 256]
        latent_vf = latent_vf.mean(dim=1)  # Giảm từ [1, 61, 256] thành [1, 256]
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        # Đảm bảo log_prob có kích thước phù hợp
        if log_prob.dim() == 1:
            log_prob = log_prob.unsqueeze(0)  # Thêm chiều batch nếu cần
        values = self.value_net(latent_vf)
        return actions, values, log_prob.sum(dim=1), lstm_states_pi  # Trả về lstm_states_pi, có thể điều chỉnh nếu cần

    def get_distribution(self, obs: th.Tensor, lstm_states: Tuple[th.Tensor, ...], episode_starts: th.Tensor) -> Tuple[Distribution, Tuple[th.Tensor, ...]]:
        features = self.extract_features(obs, lstm_states, episode_starts)
        latent_pi, lstm_states = self._process_sequence(features, lstm_states, episode_starts, self.xlstm_actor, self.context_length)
        latent_pi = latent_pi.mean(dim=1)  # Giảm chiều
        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        return self._get_action_dist_from_latent(latent_pi), lstm_states

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor, lstm_states: Tuple[th.Tensor, ...], episode_starts: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        features = self.extract_features(obs, lstm_states, episode_starts)
        if self.share_features_extractor:
            pi_features = vf_features = features
        else:
            pi_features, vf_features = features
        latent_pi, _ = self._process_sequence(pi_features, lstm_states, episode_starts, self.xlstm_actor, self.context_length)
        if self.xlstm_critic is not None and not self.shared_lstm:
            latent_vf, _ = self._process_sequence(vf_features, lstm_states, episode_starts, self.xlstm_critic, self.context_length)
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(vf_features)

        latent_pi = latent_pi.mean(dim=1)  # Giảm chiều
        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = latent_vf.mean(dim=1)  # Giảm chiều
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)

        # Đảm bảo log_prob có kích thước phù hợp
        if log_prob.dim() == 1:
            log_prob = log_prob.unsqueeze(0)  # Thêm chiều batch nếu cần
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    def _predict(self, observation: th.Tensor, lstm_states: Tuple[th.Tensor, ...], episode_starts: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, Tuple[th.Tensor, ...]]:
        distribution, lstm_states = self.get_distribution(observation, lstm_states, episode_starts)
        return distribution.get_actions(deterministic=deterministic), lstm_states

    def predict(self, observation: Union[np.ndarray, dict[str, np.ndarray]], state: Optional[Tuple[np.ndarray, ...]] = None, episode_start: Optional[np.ndarray] = None, deterministic: bool = False) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        self.set_training_mode(False)

        observation, vectorized_env = self.obs_to_tensor(observation)

        if isinstance(observation, dict):
            n_envs = observation[next(iter(observation.keys()))].shape[0]
        else:
            n_envs = observation.shape[0]
        if state is None:
            state = np.concatenate([np.zeros(self.lstm_hidden_state_shape) for _ in range(n_envs)], axis=1)
            state = (state, state)

        if episode_start is None:
            episode_start = np.array([False for _ in range(n_envs)])

        with th.no_grad():
            states = (th.tensor(state[0], dtype=th.float32, device=self.device),
                     th.tensor(state[1], dtype=th.float32, device=self.device))
            episode_starts = th.tensor(episode_start, dtype=th.float32, device=self.device)
            actions, states = self._predict(observation, lstm_states=states, episode_starts=episode_starts, deterministic=deterministic)
            states = (states[0].cpu().numpy(), states[1].cpu().numpy())

        actions = actions.cpu().numpy()

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                actions = self.unscale_action(actions)
            else:
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        if not vectorized_env:
            actions = actions.squeeze(axis=0)

        return actions, states

    def predict_values(self, obs: th.Tensor, lstm_states: Tuple[th.Tensor, ...], episode_starts: th.Tensor) -> th.Tensor:
        """
        Predict the value for a given observation using the critic network.
        Args:
            obs (th.Tensor): Observation tensor
            lstm_states (Tuple[th.Tensor, ...]): LSTM hidden and cell states
            episode_starts (th.Tensor): Indicates the start of episodes
        Returns:
            th.Tensor: Predicted value
        """
        features = self.extract_features(obs, lstm_states, episode_starts)
        if not isinstance(features, tuple):
            features = (features, features)
        _, vf_features = features

        if self.xlstm_critic is not None and not self.shared_lstm:
            latent_vf, _ = self._process_sequence(vf_features, lstm_states, episode_starts, self.xlstm_critic, self.context_length)
        elif self.shared_lstm:
            latent_vf, _ = self._process_sequence(vf_features, lstm_states, episode_starts, self.xlstm_actor, self.context_length)
        else:
            latent_vf = self.critic(vf_features)

        latent_vf = latent_vf.mean(dim=1)  # Giảm từ [1, 61, 256] thành [1, 256]
        if self.mlp_extractor is not None:
            latent_vf = self.mlp_extractor.forward_critic(latent_vf)
        values = self.value_net(latent_vf)
        return values

    def predict_values(self, obs: th.Tensor, lstm_states: Tuple[th.Tensor, ...], episode_starts: th.Tensor) -> th.Tensor:
        """
        Predict the value for a given observation using the critic network.
        Args:
            obs (th.Tensor): Observation tensor
            lstm_states (Tuple[th.Tensor, ...]): LSTM hidden and cell states
            episode_starts (th.Tensor): Indicates the start of episodes
        Returns:
            th.Tensor: Predicted value
        """
        features = self.extract_features(obs, lstm_states, episode_starts)
        if not isinstance(features, tuple):
            features = (features, features)
        _, vf_features = features

        if self.xlstm_critic is not None and not self.shared_lstm:
            latent_vf, _ = self._process_sequence(vf_features, lstm_states, episode_starts, self.xlstm_critic, self.context_length)
        elif self.shared_lstm:
            latent_vf, _ = self._process_sequence(vf_features, lstm_states, episode_starts, self.xlstm_actor, self.context_length)
        else:
            latent_vf = self.critic(vf_features)

        latent_vf = latent_vf.mean(dim=1)  # Giảm từ [1, 61, 256] thành [1, 256]
        if self.mlp_extractor is not None:
            latent_vf = self.mlp_extractor.forward_critic(latent_vf)
        values = self.value_net(latent_vf)
        return values

class RecurrentActorCriticCnnPolicy(RecurrentActorCriticPolicy):
    """
    CNN recurrent policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param lstm_hidden_size: Number of hidden units for each LSTM layer.
    :param n_lstm_layers: Number of LSTM layers.
    :param shared_lstm: Whether the LSTM is shared between the actor and the critic.
        By default, only the actor has a recurrent network.
    :param enable_critic_lstm: Use a seperate LSTM for the critic.
    :param lstm_kwargs: Additional keyword arguments to pass the the LSTM
        constructor.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 1,
        shared_lstm: bool = False,
        enable_critic_lstm: bool = True,
        lstm_kwargs: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            lstm_hidden_size,
            n_lstm_layers,
            shared_lstm,
            enable_critic_lstm,
            lstm_kwargs,
        )


class RecurrentMultiInputActorCriticPolicy(RecurrentActorCriticPolicy):
    """
    MultiInputActorClass policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param lstm_hidden_size: Number of hidden units for each LSTM layer.
    :param n_lstm_layers: Number of LSTM layers.
    :param shared_lstm: Whether the LSTM is shared between the actor and the critic.
        By default, only the actor has a recurrent network.
    :param enable_critic_lstm: Use a seperate LSTM for the critic.
    :param lstm_kwargs: Additional keyword arguments to pass the the LSTM
        constructor.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 1,
        shared_lstm: bool = False,
        enable_critic_lstm: bool = True,
        lstm_kwargs: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            lstm_hidden_size,
            n_lstm_layers,
            shared_lstm,
            enable_critic_lstm,
            lstm_kwargs,
        )




