import json
import os
import time
from typing import Sequence, Tuple
import numpy as np
import jax
import jax.numpy as jnp
from jaxued.environments.underspecified_env import EnvParams, EnvState, UnderspecifiedEnv
from jaxued.utils import compute_max_mean_returns_epcount, compute_max_mean_returns_epcount_w_idxs
import optax
from flax import struct
from flax.training.train_state import TrainState as BaseTrainState
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import distrax
import orbax.checkpoint as ocp
import wandb
from jaxued.environments.maze.env_editor import MazeEditor, Observation, ObservedMazeEditor, ObservedMazeEditorWithGoal
from jaxued.linen import ResetRNN
from jaxued.environments import Maze, MazeRenderer, ObservedMazeRenderer
from jaxued.environments.maze import Level, ObservedLevel
from jaxued.wrappers import AutoReplayWrapper
import chex

import logging
import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

# region PPO helper functions    
@struct.dataclass
class TrainState:
    update_count: int
    pro_train_state: BaseTrainState
    adv_train_state: BaseTrainState

def compute_gae(
    gamma: float,
    lambd: float,
    last_value: chex.Array,
    values: chex.Array,
    rewards: chex.Array,
    dones: chex.Array,
    traj_idxs: chex.Array
) -> Tuple[chex.Array, chex.Array]:
    """This takes in arrays of shape (NUM_STEPS, NUM_ENVS) and returns the advantages and targets.

    Args:
        gamma (float): 
        lambd (float): 
        last_value (chex.Array):  Shape (NUM_ENVS)
        values (chex.Array): Shape (NUM_STEPS, NUM_ENVS)
        rewards (chex.Array): Shape (NUM_STEPS, NUM_ENVS)
        dones (chex.Array): Shape (NUM_STEPS, NUM_ENVS)

    Returns:
        Tuple[chex.Array, chex.Array]: advantages, targets; each of shape (NUM_STEPS, NUM_ENVS)
    """
    def compute_gae_at_timestep(carry, x):
        gae, next_value, cur_idx = carry
        traj_started = cur_idx <= traj_idxs
    
        value, reward, done = x
        delta = reward + gamma * next_value * (1 - done) * traj_started - value
        gae = delta + gamma * lambd * (1 - done) * gae
        return (gae, jnp.where(traj_started, value, next_value), cur_idx - 1), gae

    _, advantages = jax.lax.scan(
        compute_gae_at_timestep,
        (jnp.zeros_like(last_value), last_value, values.shape[0]),
        (values, rewards, dones),
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + values

def sample_trajectories(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    adv_env: UnderspecifiedEnv,
    adv_env_params: EnvParams,
    train_state: TrainState,
    init_levels: ObservedLevel,
    num_envs: int,
    max_episode_length: int,
    max_adv_episode_length: int,
):

    pro_train_state = train_state.pro_train_state
    adv_train_state = train_state.adv_train_state


    rng, _rng = jax.random.split(rng)
    adv_init_obs, adv_init_env_state = jax.vmap(adv_env.reset_to_level, in_axes=(0, 0, None))(jax.random.split(_rng, num_envs), init_levels, adv_env_params)
    init_student_obs_mask = jnp.ones((num_envs, adv_env.action_space(adv_env_params).n), dtype=jnp.bool_)

    # Create Padded Trajectory accounting for max size of adversary trajectory
    adv_init_traj = (
        adv_init_obs, 
        jnp.zeros(num_envs, dtype=jnp.int32),
        jnp.zeros(num_envs, dtype=jnp.int32),
        jnp.zeros(num_envs, dtype=jnp.bool_),
        jnp.zeros(num_envs, dtype=jnp.float32),
        jnp.zeros(num_envs, dtype=jnp.float32),
        {}
    )

    def pad_traj(input_array: chex.Array):
        return jnp.stack([input_array] + [jnp.zeros_like(input_array)] * (max_adv_episode_length - 1))

    adv_traj_idxs = jnp.zeros(num_envs, dtype=jnp.int32)
    adv_init_traj = jax.tree_map(pad_traj, adv_init_traj)

    # Student Step in Env
    def sample_student_step(carry, _):
        rng, train_state, hstate, obs, env_state, last_done = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        x = jax.tree_map(lambda x: x[None, ...], (obs, last_done))
        hstate, pi, value = train_state.apply_fn(train_state.params, x, hstate)
        action = pi.sample(seed=rng_action)
        log_prob = pi.log_prob(action)
        value, action, log_prob = (
            value.squeeze(0),
            action.squeeze(0),
            log_prob.squeeze(0),
        )

        next_obs, env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_envs), env_state, action, env_params)

        carry = (rng, train_state, hstate, next_obs, env_state, done)
        return carry, (obs, action, reward, done, log_prob, value, info)

    # Adversary Step in Env
    def sample_adv_step(carry):
        # Env Step that Updates Action Mask
        def env_step(rng, state, action, params, student_obs_mask, student_observations):
            next_obs, env_state, reward, done, info = adv_env.step(rng, state, action, params)
            action_mask = jnp.logical_and(next_obs.action_mask, student_obs_mask)
            return next_obs.replace(action_mask=action_mask, agent_observations=student_observations, place_goal=jnp.array(False, dtype=jnp.bool_)), env_state, reward, done, info

        rng, train_state, hstate, obs, env_state, last_done, student_obs_mask, student_observations, adv_full_traj, adv_traj_idxs = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        # Check which envs still have remaining actions
        keep_gen = ~jnp.all(~obs.action_mask, axis=1)
        
        traj_not_full = adv_traj_idxs < max_adv_episode_length
        keep_gen = jnp.logical_and(keep_gen, traj_not_full)

        x = jax.tree_map(lambda x: x[None, ...], (obs, last_done))
        next_hstate, pi, value = train_state.apply_fn(train_state.params, x, hstate)
        action = pi.sample(seed=rng_action)
        log_prob = pi.log_prob(action)
        value, action, log_prob = (
            value.squeeze(0),
            action.squeeze(0),
            log_prob.squeeze(0),
        )

        next_obs, next_env_state, reward, done, info = jax.vmap(
            env_step, in_axes=(0, 0, 0, None, 0, 0)
        )(jax.random.split(rng_step, num_envs), env_state, action, env_params, student_obs_mask, student_observations)

        traj_step = (obs, action, reward, done, log_prob, value, info)
        
        # Add next trajectory step to the full padded traj if action was available
        def update_traj(full_traj, traj_step):
            def update_at_idx(full_traj, traj_step, idx, keep_gen):
                return jax.lax.cond(keep_gen,
                    lambda : jax.lax.dynamic_update_slice_in_dim(full_traj, traj_step[None, ...], idx, 0),
                    lambda : full_traj
                )
            
            return jax.vmap(
                update_at_idx, in_axes=(1, 0, 0, 0)
            )(full_traj, traj_step, adv_traj_idxs, keep_gen).swapaxes(1, 0)

        adv_full_traj = jax.tree_map(update_traj, adv_full_traj, traj_step)

        # Replace Carry for envs with available actions
        def replace_carry(new, old):
            expanded_keep_gen = jnp.reshape(keep_gen, (keep_gen.shape[0],) + (1,) * (new.ndim - 1))
            return jnp.where(expanded_keep_gen, new, old)
        
        (next_hstate, next_obs, next_env_state, done, adv_traj_idxs) = jax.tree_map(
            replace_carry,
            (next_hstate, next_obs, next_env_state, done, adv_traj_idxs + 1),
            (hstate, obs, env_state, last_done, adv_traj_idxs)
        )
        
        carry = (
            rng, 
            train_state, 
            next_hstate, 
            next_obs, 
            next_env_state, 
            done, 
            student_obs_mask, 
            student_observations,
            adv_full_traj, 
            adv_traj_idxs
        )
        return carry

    (
        rng, adv_train_state, adv_hstate, adv_last_obs, adv_last_env_state, adv_last_done, student_obs_mask, student_observations, adv_full_traj, adv_traj_idxs
    ) = jax.lax.while_loop(
        lambda carry : (carry[-1] < 2).all(),
        sample_adv_step,
        (
            rng,
            adv_train_state,
            AdversaryActorCritic.initialize_carry((num_envs,)),
            adv_init_obs,
            adv_init_env_state,
            jnp.zeros(num_envs, dtype=bool),
            init_student_obs_mask,
            adv_init_obs.agent_observations,
            adv_init_traj,
            adv_traj_idxs
        ),
    )

    # Initialise Student
    rng, _rng = jax.random.split(rng)
    init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(jax.random.split(_rng, num_envs), adv_last_env_state.level, env_params)

    pro_carry = (
        pro_train_state, 
        ActorCritic.initialize_carry((num_envs,)), 
        init_obs, 
        init_env_state, 
        jnp.zeros(num_envs, dtype=bool)
    )

    adv_carry = (
        adv_train_state, 
        adv_hstate, 
        adv_last_obs, 
        adv_last_env_state, 
        adv_last_done, 
        student_obs_mask, 
        adv_full_traj, 
        adv_traj_idxs
    )

    # get adversary action 
    def sample_step(carry, _):
        rng, pro_carry, adv_carry = carry

        pro_train_state, pro_hstate, pro_obs, pro_env_state, pro_last_done = pro_carry
        adv_train_state, adv_hstate, _, adv_last_env_state, adv_last_done, student_obs_mask, adv_full_traj, adv_traj_idxs = adv_carry
        adv_last_obs = jax.vmap(adv_env.get_obs, in_axes=(0, 0))(jax.random.split(_rng, num_envs), adv_last_env_state)

        student_observations = jnp.stack([pro_obs.agent_observation, pro_obs.agent_observation], axis=3)
        student_obs_mask = jnp.concatenate([jnp.logical_or(student_observations[:, :, :, 0], student_observations[:, :, :, 1]).reshape(num_envs, -1)]*3, axis=1)
        action_mask = jnp.logical_and(adv_last_obs.action_mask, student_obs_mask)

        # Prevent Goal Placing on all but the first action of the set
        student_obs_mask = jnp.logical_and(student_obs_mask, jnp.concatenate([jnp.ones((num_envs, adv_env.action_space(adv_env_params).n * 2 // 3)), jnp.zeros((num_envs, adv_env.action_space(adv_env_params).n // 3))], axis=1))

        # Get Protagonist Location
        agent_location = jnp.concatenate([pro_env_state.env_state.agent_dir[:, None], pro_env_state.env_state.agent_pos], axis=-1)

        (
            (rng, adv_train_state, adv_hstate, adv_last_obs, adv_last_env_state, adv_last_done, student_obs_mask, student_observations, adv_full_traj, adv_traj_idxs)
        ) = jax.lax.while_loop(
            lambda carry : ~jnp.logical_or(jnp.all(~carry[3].action_mask, axis=1), ~(carry[9] < max_adv_episode_length)).all(),
            #lambda carry : ~(~carry[3].action_mask).all()
            sample_adv_step,
            (
                rng,
                adv_train_state,
                adv_hstate,
                adv_last_obs.replace(action_mask=action_mask, agent_observations=student_observations),
                adv_last_env_state,
                adv_last_done,
                student_obs_mask,
                student_observations,
                adv_full_traj,
                adv_traj_idxs
            )
        )

        # update student envs
        pro_obs, pro_env_state = jax.vmap(env.update_state_from_level, in_axes=(0, 0))(adv_last_env_state.level, pro_env_state)

        # students act in env
        full_pro_carry, pro_traj_step = sample_student_step(
            (rng, pro_train_state, pro_hstate, pro_obs, pro_env_state, pro_last_done), 
            None
        )
        rng, pro_carry = full_pro_carry[0], full_pro_carry[1:]

        # Return Carry
        adv_carry = (adv_train_state, adv_hstate, adv_last_obs, adv_last_env_state, adv_last_done, student_obs_mask, adv_full_traj, adv_traj_idxs)
        carry = (rng, pro_carry, adv_carry)

        return carry, (pro_traj_step, adv_traj_idxs, agent_location)

    (rng, pro_carry, adv_carry), (pro_traj, adv_traj_idx_array, pro_locations) = jax.lax.scan(
        sample_step, 
        (rng, pro_carry, adv_carry), 
        None, 
        length=max_episode_length
    )

    def get_last_value(carry):
        train_state, hstate, last_obs, _, last_done = carry[:5]
        x = jax.tree_map(lambda x: x[None, ...], (last_obs, last_done))
        _, _, last_value = train_state.apply_fn(train_state.params, x, hstate)
        return last_value

    last_values = get_last_value(pro_carry).squeeze(0), get_last_value(adv_carry).squeeze(0)

    last_pro_env_state = pro_carry[3]
    last_agent_location = jnp.concatenate([last_pro_env_state.env_state.agent_dir[:, None], last_pro_env_state.env_state.agent_pos], axis=-1)    

    return rng, (pro_carry[:4], adv_carry[:4]), (pro_traj, adv_carry[6:]), last_values, adv_traj_idx_array, jnp.concatenate([pro_locations, last_agent_location[None, ...]], axis=0)

def evaluate_rnn(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state: TrainState,
    init_hstate: chex.ArrayTree,
    init_obs: Observation,
    init_env_state: EnvState,
    max_episode_length: int,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """This runs the RNN on the environment, given an initial state and observation, and returns (states, rewards, episode_lengths)

    Args:
        rng (chex.PRNGKey): 
        env (UnderspecifiedEnv): 
        env_params (EnvParams): 
        train_state (TrainState): 
        init_hstate (chex.ArrayTree): Shape (num_levels, )
        init_obs (Observation): Shape (num_levels, )
        init_env_state (EnvState): Shape (num_levels, )
        max_episode_length (int): 

    Returns:
        Tuple[chex.Array, chex.Array, chex.Array]: (States, rewards, episode lengths) ((NUM_STEPS, NUM_LEVELS), (NUM_STEPS, NUM_LEVELS), (NUM_LEVELS,)
    """
    num_levels = jax.tree_util.tree_flatten(init_obs)[0][0].shape[0]
    
    def step(carry, _):
        rng, hstate, obs, state, done, mask, episode_length = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        x = jax.tree_map(lambda x: x[None, ...], (obs, done))
        hstate, pi, _ = train_state.apply_fn(train_state.params, x, hstate)
        action = pi.sample(seed=rng_action).squeeze(0)

        obs, next_state, reward, done, _ = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_levels), state, action, env_params)
        
        next_mask = mask & ~done
        episode_length += mask

        return (rng, hstate, obs, next_state, done, next_mask, episode_length), (state, reward)
    
    (_, _, _, _, _, _, episode_lengths), (states, rewards) = jax.lax.scan(
        step,
        (
            rng,
            init_hstate,
            init_obs,
            init_env_state,
            jnp.zeros(num_levels, dtype=bool),
            jnp.ones(num_levels, dtype=bool),
            jnp.zeros(num_levels, dtype=jnp.int32),
        ),
        None,
        length=max_episode_length,
    )

    return states, rewards, episode_lengths

def masked_mean(arr, mask):
    return jnp.sum(arr * mask) / jnp.sum(mask)

def masked_std(arr, mask):
    mean = masked_mean(arr, mask)
    var = jnp.sum(mask*(arr - mean)**2)/jnp.sum(mask)
    return jnp.sqrt(var)

def update_actor_critic_rnn(
    rng: chex.PRNGKey,
    train_state: TrainState,
    init_hstate: chex.ArrayTree,
    batch: chex.ArrayTree,
    num_envs: int,
    n_steps: int,
    n_minibatch: int,
    n_epochs: int,
    clip_eps: float,
    entropy_coeff: float,
    critic_coeff: float,
    update_grad: bool=True,
) -> Tuple[Tuple[chex.PRNGKey, TrainState], chex.ArrayTree]:
    """This function takes in a rollout, and PPO hyperparameters, and updates the train state.

    Args:
        rng (chex.PRNGKey): 
        train_state (TrainState): 
        init_hstate (chex.ArrayTree): 
        batch (chex.ArrayTree): obs, actions, dones, log_probs, values, targets, advantages
        num_envs (int): 
        n_steps (int): 
        n_minibatch (int): 
        n_epochs (int): 
        clip_eps (float): 
        entropy_coeff (float): 
        critic_coeff (float): 
        update_grad (bool, optional): If False, the train state does not actually get updated. Defaults to True.

    Returns:
        Tuple[Tuple[chex.PRNGKey, TrainState], chex.ArrayTree]: It returns a new rng, the updated train_state, and the losses. The losses have structure (loss, (l_vf, l_clip, entropy))
    """
    obs, actions, dones, log_probs, values, targets, advantages, update_mask = batch
    last_dones = jnp.roll(dones, 1, axis=0).at[0].set(False)
    batch = obs, actions, last_dones, log_probs, values, targets, advantages, update_mask
    
    def update_epoch(carry, _):
        def update_minibatch(train_state, minibatch):
            init_hstate, obs, actions, last_dones, log_probs, values, targets, advantages, update_mask = minibatch
            
            def loss_fn(params):
                _, pi, values_pred = train_state.apply_fn(params, (obs, last_dones), init_hstate)
                log_probs_pred = pi.log_prob(actions)
                entropy = masked_mean(pi.entropy(), update_mask)

                ratio = jnp.exp(log_probs_pred - log_probs)
                A = (advantages - masked_mean(advantages, update_mask)) / (masked_std(advantages, update_mask) + 1e-5)
                l_clip = masked_mean((-jnp.minimum(ratio * A, jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * A)), update_mask)

                values_pred_clipped = values + (values_pred - values).clip(-clip_eps, clip_eps)
                l_vf = masked_mean(0.5 * jnp.maximum((values_pred - targets) ** 2, (values_pred_clipped - targets) ** 2), update_mask)

                loss = l_clip + critic_coeff * l_vf - entropy_coeff * entropy

                return loss, (l_vf, l_clip, entropy)

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            loss, grads = grad_fn(train_state.params)
            if update_grad:
                train_state = train_state.apply_gradients(grads=grads)
            return train_state, loss

        rng, train_state = carry
        rng, rng_perm = jax.random.split(rng)
        permutation = jax.random.permutation(rng_perm, num_envs)
        minibatches = (
            jax.tree_map(
                lambda x: jnp.take(x, permutation, axis=0)
                .reshape(n_minibatch, -1, *x.shape[1:]),
                init_hstate,
            ),
            *jax.tree_map(
                lambda x: jnp.take(x, permutation, axis=1)
                .reshape(x.shape[0], n_minibatch, -1, *x.shape[2:])
                .swapaxes(0, 1),
                batch,
            ),
        )
        train_state, losses = jax.lax.scan(update_minibatch, train_state, minibatches)
        return (rng, train_state), losses

    return jax.lax.scan(update_epoch, (rng, train_state), None, n_epochs)

def compute_min_steps_to_goal(level):
    #wall_values = jnp.repeat(jnp.where(level.wall_map, jnp.inf, -jnp.inf)[None, ...], 4, axis=0) # unseen squares treated as empty
    wall_values = jnp.repeat(jnp.where(jnp.logical_or(level.wall_map, ~level.observation_map), jnp.inf, -jnp.inf)[None, ...], 4, axis=0) # unseen squares treated as walls
    max_height, max_width = level.wall_map.shape
    
    def compute_next(values):
        fwd_values = jnp.array([
            jnp.roll(values[0], -1, axis=1).astype(float).at[:,-1].set(jnp.inf),
            jnp.roll(values[1], -1, axis=0).astype(float).at[-1,:].set(jnp.inf),
            jnp.roll(values[2], 1, axis=1).astype(float).at[:,0].set(jnp.inf),
            jnp.roll(values[3], 1, axis=0).astype(float).at[0,:].set(jnp.inf),
        ])
        new_values = jnp.empty_like(values)
        for i in range(4):
            new_values = new_values.at[i].set(jnp.min(
                jnp.array([values[i], values[i-1] + 1, values[(i+1)%4] + 1, fwd_values[i] + 1]), axis=0
            ))
        return jnp.maximum(new_values, wall_values)
    
    def cond_fn(carry):
        values, next_values = carry
        return jnp.any(values != next_values)
    
    def body_fn(carry):
        _, values = carry
        return values, compute_next(values)
    
    values = jnp.full((4, max_height, max_width), jnp.inf)
    values = jax.lax.select(level.goal_placed, values.at[:, level.goal_pos[1], level.goal_pos[0]].set(0), values)
    return jax.lax.while_loop(cond_fn, body_fn, (values, compute_next(values)))[0]

class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    
    @nn.compact
    def __call__(self, inputs, hidden):
        obs, dones = inputs
        
        img_embed = nn.Conv(16, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(obs.image)
        img_embed = img_embed.reshape(*img_embed.shape[:-3], -1)
        img_embed = nn.relu(img_embed)
        
        dir_embed = jax.nn.one_hot(obs.agent_dir, 4)
        dir_embed = nn.Dense(5, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="scalar_embed")(dir_embed)
        
        embedding = jnp.append(img_embed, dir_embed, axis=-1)

        hidden, embedding = ResetRNN(nn.OptimizedLSTMCell(features=256))((embedding, dones), initial_carry=hidden)

        actor_mean = nn.Dense(32, kernel_init=orthogonal(2), bias_init=constant(0.0), name="actor0")(embedding)
        actor_mean = nn.tanh(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="actor1")(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(32, kernel_init=orthogonal(2), bias_init=constant(0.0), name="critic0")(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="critic1")(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)
    
    @staticmethod
    def initialize_carry(batch_dims):
        return nn.OptimizedLSTMCell(features=256).initialize_carry(jax.random.PRNGKey(0), (*batch_dims, 256))


class AdversaryActorCritic(nn.Module):
    # The adversary's network architecture
    action_dim: Sequence[int]
    max_timesteps: int = 50
    
    @nn.compact
    def __call__(self, inputs: Tuple[Observation, chex.Array], hidden):
        obs, dones = inputs
        
        img_embed = nn.Conv(128, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(jnp.concatenate((obs.image, jnp.expand_dims(obs.observation_map, axis=-1), obs.agent_observations), axis=-1))
        img_embed = img_embed.reshape(*img_embed.shape[:-3], -1)
        img_embed = nn.relu(img_embed)
        
        time_value = nn.Embed(self.max_timesteps + 1, 10, name="time_embed", embedding_init=orthogonal(1.0))(jnp.clip(obs.time, None, self.max_timesteps))
        random_z_value = obs.random_z
        embedding = jnp.concatenate((img_embed, time_value, random_z_value), axis=-1)

        hidden, embedding = ResetRNN(nn.OptimizedLSTMCell(features=256))((embedding, dones), initial_carry=hidden)

        actor_mean = nn.Dense(32, kernel_init=orthogonal(2), bias_init=constant(0.0), name="actor0")(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="actor1")(actor_mean)

        # Mask out this
        actor_mean = jnp.where(obs.action_mask, actor_mean, -jnp.inf)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(32, kernel_init=orthogonal(2), bias_init=constant(0.0), name="critic0")(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="critic1")(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)
    
    @staticmethod
    def initialize_carry(batch_dims):
        return nn.OptimizedLSTMCell(features=256).initialize_carry(jax.random.PRNGKey(0), (*batch_dims, 256))
# endregion

# region checkpointing
def setup_checkpointing(config: dict, train_state: TrainState, env: UnderspecifiedEnv, env_params: EnvParams) -> ocp.CheckpointManager:
    """This takes in the train state and config, and returns an orbax checkpoint manager.
        It also saves the config in `checkpoints/run_name/seed/config.json`

    Args:
        config (dict): 
        train_state (TrainState): 
        env (UnderspecifiedEnv): 
        env_params (EnvParams): 

    Returns:
        ocp.CheckpointManager: 
    """
    overall_save_dir = os.path.join(os.getcwd(), "obs_generation/checkpoints", f"{config['run_name']}", str(config['seed']))
    os.makedirs(overall_save_dir, exist_ok=True)
    
    # save the config
    config_dict = OmegaConf.to_container(config, resolve=True)
    with open(os.path.join(overall_save_dir, 'config.json'), 'w+') as f:
        f.write(json.dumps(config_dict, indent=True))
    
    checkpoint_manager = ocp.CheckpointManager(
        os.path.join(overall_save_dir, 'models'),
        options=ocp.CheckpointManagerOptions(
            save_interval_steps=config['checkpoint_save_interval'],
            max_to_keep=config['max_number_of_checkpoints'],
            enable_async_checkpointing=False,
        )
    )
    
    return checkpoint_manager
#endregion

def main(config=None, project="JAXUED_TEST"):
    wandb_config = OmegaConf.to_container(
            config, resolve=True, throw_on_missing=False
        )
    
    #run = wandb.init(config=config, project=project, group=config["group_name"], tags=["PAIRED",])
    run = wandb.init(config=wandb_config, project=project, tags=["PVL", "obs_gen",])
    
    wandb.define_metric("num_updates")
    wandb.define_metric("num_env_steps")
    wandb.define_metric("solve_rate/*", step_metric="num_updates")
    wandb.define_metric("level_sampler/*", step_metric="num_updates")
    wandb.define_metric("agent/*", step_metric="num_updates")
    wandb.define_metric("misc/*", step_metric="num_updates")
    wandb.define_metric("return/*", step_metric="num_updates")
    wandb.define_metric("eval_ep_length/*", step_metric="num_updates")

    def log_eval(stats):
        logger.info(f"Logging update: {stats['update_count']}")
        
        # generic stats
        env_steps = 2 * stats["update_count"] * config["num_train_envs"] * config["student_num_steps"]
        log_dict = {
            "misc/mean_num_blocks": stats["mean_num_blocks"].mean(),
            "num_updates": stats["update_count"],
            "num_env_steps": env_steps,
            "sps": env_steps / stats['time_delta'],
            "misc/prot_perf_mean": stats['pro_mean_returns'].mean(),
            "misc/prot_perf_max":  stats['pro_max_returns'].mean(),
            "misc/pro_num_episodes": stats['pro_eps'].mean(),
            "misc/optimal_regret":   stats['optimal_regret'].mean(),
            "misc/positive_value_loss":   stats['positive_value_loss'].mean(),
        }
        
        # evaluation performance
        solve_rates = stats['eval_solve_rates']
        returns     = stats["eval_returns"]
        log_dict.update({f"solve_rate/{name}": solve_rate for name, solve_rate in zip(config["eval_levels"], solve_rates)})
        log_dict.update({"solve_rate/mean": solve_rates.mean()})
        log_dict.update({f"return/{name}": ret for name, ret in zip(config["eval_levels"], returns)})
        log_dict.update({"return/mean": returns.mean()})
        log_dict.update({"eval_ep_lengths/mean": stats['eval_ep_lengths'].mean()})
        def make_caption(i):
            pro_mean_returns = jnp.round(stats['pro_mean_returns'][-1][i], 2) # .flatten()
            optimal_regret       = jnp.round(stats['optimal_regret'][-1][i], 2) # .flatten()
            return f"P({pro_mean_returns:.2f})|R({optimal_regret:.2f})"
    
        log_dict.update({f"images/levels": [wandb.Image(np.array(image), caption=make_caption(i)) for i, image in enumerate(stats["levels"])]})

        # animations
        for i, level_name in enumerate(config["eval_levels"]):
            frames, episode_length = stats["eval_animation"][0][:, i], stats["eval_animation"][1][i]
            frames = np.array(frames[:episode_length])
            log_dict.update({f"animations/{level_name}": wandb.Video(frames, fps=4)})
        
        wandb.log(log_dict)
    
    env = Maze(max_height=13, max_width=13, agent_view_size=5, normalize_obs=True)
    adv_env = ObservedMazeEditorWithGoal(env, random_z_dimensions=config['adv_random_z_dimension'], zero_out_random_z=config['adv_zero_out_random_z'])
    eval_env = env
    adv_env_renderer = ObservedMazeRenderer(env, tile_size=8)
    env_renderer = MazeRenderer(env, tile_size=8)
    env = AutoReplayWrapper(env)
    env_params = env.default_params
    adv_env_params = adv_env.default_params

    def sample_empty_level():
        w, h = env._env.max_width, env._env.max_height
        return ObservedLevel(
            wall_map=jnp.zeros((h, w), dtype=jnp.bool_),
            observation_map=jnp.zeros((h, w), dtype=jnp.bool_),
            width=w,
            height=h,
            
            # These values don't matter, as the adversary overwrites them.
            goal_pos=jnp.array([0, 0], dtype=jnp.uint32),
            agent_pos=jnp.array([1, 1], dtype=jnp.uint32),
            agent_dir=jnp.array(0, dtype=jnp.uint8),
            goal_placed=jnp.array(False, dtype=jnp.bool_),
        )

    def create_train_state(rng):
        def create_inner_train_state(rng, env, env_params, network_cls, prefix, network_kws={}):
            def linear_schedule(count):
                frac = (
                    1.0
                    - (count // (config[f"{prefix}num_minibatches"] * config[f"{prefix}epoch_ppo"]))
                    / config["num_updates"]
                )
                return config[f"{prefix}lr"] * frac
            obs, _ = env.reset_to_level(rng, sample_empty_level(), env_params)
            obs = jax.tree_map(
                lambda x: jnp.repeat(jnp.repeat(x[None, ...], config["num_train_envs"], axis=0)[None, ...], 256, axis=0),
                obs,
            )
            init_x = (obs, jnp.zeros((256, config["num_train_envs"])))
            network = network_cls(env.action_space(env_params).n, **network_kws)
            network_params = network.init(rng, init_x, network_cls.initialize_carry((config["num_train_envs"],)))
            tx = optax.chain(
                optax.clip_by_global_norm(config[f"{prefix}max_grad_norm"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
                # optax.adam(learning_rate=config[f"{prefix}lr"], eps=1e-5),
            )
            return BaseTrainState.create(
                apply_fn=network.apply,
                params=network_params,
                tx=tx,
            )
        rng_pro, rng_ant, rng_adv = jax.random.split(rng, 3)
        return TrainState(
            update_count = 0,
            pro_train_state = create_inner_train_state(rng_pro, env, env_params, ActorCritic, "student_"),
            adv_train_state = create_inner_train_state(rng_adv, adv_env, adv_env_params, AdversaryActorCritic, "adv_", network_kws={"max_timesteps": config["adv_num_steps"]})
        )

    def train_step(carry, _):
        def get_rollout(traj, traj_idxs, last_value, prefix):
            obs, actions, rewards, dones, log_probs, values, info = traj
            advantages, targets = compute_gae(config[f"{prefix}gamma"], config[f"{prefix}gae_lambda"], last_value, values, rewards, dones, traj_idxs)
            update_mask = jnp.tile(jnp.arange(actions.shape[0])[:, None], (1, actions.shape[1])) < traj_idxs
            return (obs, actions, dones, log_probs, values, targets, advantages, update_mask), (dones, rewards)
        
        def update(rng, train_state, init_hstate, rollout, prefix):
            # Returns: (rng, train_state), losses
            return update_actor_critic_rnn(
                rng,
                train_state,
                init_hstate,
                rollout,
                config["num_train_envs"],
                config[f"{prefix}num_steps"],
                config[f"{prefix}num_minibatches"],
                config[f"{prefix}epoch_ppo"],
                config[f"{prefix}clip_eps"],
                config[f"{prefix}entropy_coeff"],
                config[f"{prefix}critic_coeff"],
                update_grad=True,
            )
        
        def get_agent_min_steps_to_goal(env_state, agent_locations):
            all_optimal_distances = compute_min_steps_to_goal(env_state.level).swapaxes(1, 2)

            agent_optimal_distances = jax.vmap(
                lambda x : all_optimal_distances[tuple(x)],
                in_axes=(0)
            )(agent_locations)

            return agent_optimal_distances

        def add_at_idx(arr, add, idx):
            old_val = jax.lax.dynamic_index_in_dim(arr, idx-1, axis=0)
            return jax.lax.dynamic_update_index_in_dim(arr, add + old_val, idx-1, axis=0)

        def add_regret_at_idx(adv_rewards, input):
            regret, idxs = input
            adv_rewards = jax.vmap(add_at_idx, in_axes=(1, 0, 0))(adv_rewards, regret, idxs).swapaxes(0, 1)
            return adv_rewards, None
        
        rng, train_state = carry
        
        # Initialise Levels
        empty_levels = jax.tree_map(lambda x: jnp.array([x]).repeat(config["num_train_envs"], axis=0), sample_empty_level())

        # Gather Trajectories
        rng, (pro_carry, adv_carry), (pro_traj, (adv_traj, adv_traj_idxs)), (pro_last_value, adv_last_value), adv_traj_idx_array, pro_locations = sample_trajectories(
            rng,
            env,
            env_params,
            adv_env,
            adv_env_params,
            train_state,
            empty_levels,
            config["num_train_envs"],
            config["student_num_steps"],
            config["adv_num_steps"]
        )
        
        # Get Rollouts for Protagonist and Antagonist
        pro_rollout, (dones, rewards) = get_rollout(pro_traj, jnp.ones(config["num_train_envs"], dtype=jnp.int32) * config["student_num_steps"], pro_last_value, 'student_')
        pro_mean_returns, pro_max_returns, pro_eps = compute_max_mean_returns_epcount(dones, rewards)

        # Set Adversary Reward and Done Arrays
        obs, actions, rewards, dones, log_probs, values, info = adv_traj
        dones = jax.vmap(
            lambda x, i : jax.lax.dynamic_update_slice(x, jnp.ones((1,), dtype=jnp.bool_), (i-1,)),
            in_axes=(1, 0)
        )(dones, adv_traj_idxs).swapaxes(0, 1)

        # Get Optimal Distances
        agent_optimal_distances = jax.vmap(
            get_agent_min_steps_to_goal,
            in_axes=(0, 1)
        )(adv_carry[3], pro_locations).swapaxes(0, 1)

        # Compute one-step pro regret
        agent_regret = jnp.nan_to_num((1 + agent_optimal_distances[1:]) - (agent_optimal_distances[:-1]), 0) * (1 - pro_traj[3]) / pro_eps

        # Get PVL
        pro_advantage = pro_rollout[6]
        positive_value_loss = jnp.maximum(pro_advantage, 0)

        # Assign PVL to Adv
        rewards, _ = jax.lax.scan(
            add_regret_at_idx,
            jnp.zeros(rewards.shape),
            (positive_value_loss, adv_traj_idx_array),
        )

        rewards /= config["student_num_steps"] # Normalise Regret

        adv_traj = obs, actions, rewards, dones, log_probs, values, info
        adv_rollout, _ = get_rollout(adv_traj, adv_traj_idxs, adv_last_value, 'adv_')

        (rng, pro_train_state), pro_losses = update(rng, train_state.pro_train_state, ActorCritic.initialize_carry((config["num_train_envs"],)), pro_rollout, "student_")
        (rng, adv_train_state), adv_losses = update(rng, train_state.adv_train_state, AdversaryActorCritic.initialize_carry((config["num_train_envs"],)), adv_rollout, "adv_")

        _, _, _, adv_last_env_state = adv_carry
        levels = adv_last_env_state.level

        metrics = {
            "pro_losses": jax.tree_map(lambda x: x.mean(), pro_losses),
            "adv_losses": jax.tree_map(lambda x: x.mean(), adv_losses),
            "mean_num_blocks": levels.wall_map.sum() / config["num_train_envs"],
            "pro_mean_returns": pro_mean_returns,
            "pro_max_returns":  pro_max_returns,
            "optimal_regret":   agent_regret.sum(axis=0),
            "positive_value_loss": positive_value_loss.sum(axis=0),
            "pro_eps": pro_eps,
            "levels": levels,
        }

        train_state = train_state.replace(
            update_count=train_state.update_count + 1,
            pro_train_state=pro_train_state,
            adv_train_state=adv_train_state,
        )
        return (rng, train_state), metrics
    
    def eval(rng, train_state):
        rng, rng_reset = jax.random.split(rng)
        levels = Level.load_prefabs(config["eval_levels"])
        num_levels = len(config["eval_levels"])
        init_obs, init_env_state = jax.vmap(eval_env.reset_to_level, (0, 0, None))(jax.random.split(rng_reset, num_levels), levels, env_params)
        states, rewards, episode_lengths = evaluate_rnn(
            rng,
            eval_env,
            env_params,
            train_state,
            ActorCritic.initialize_carry((num_levels,)),
            init_obs,
            init_env_state,
            env_params.max_steps_in_episode,
        )
        mask = jnp.arange(env_params.max_steps_in_episode)[..., None] < episode_lengths
        cum_rewards = (rewards * mask).sum(axis=0)
        return states, cum_rewards, episode_lengths # (num_steps, num_eval_levels, ...), (num_eval_levels,), (num_eval_levels,)
    
    @jax.jit
    def train_and_eval_step(runner_state, _):
        (rng, train_state), metrics = jax.lax.scan(train_step, runner_state, None, config["eval_freq"])
        
        rng, rng_eval = jax.random.split(rng)
        states, cum_rewards, episode_lengths = jax.vmap(eval, (0, None))(jax.random.split(rng_eval, config["eval_num_attempts"]), train_state.pro_train_state)
        eval_solve_rates = jnp.where(cum_rewards > 0, 1., 0.).mean(axis=0) # (num_eval_levels,)
        eval_returns = cum_rewards.mean(axis=0) # (num_eval_levels,)
        
        # just grab the first run
        states, episode_lengths = jax.tree_map(lambda x: x[0], (states, episode_lengths)) # (num_steps, num_eval_levels, ...), (num_eval_levels,)
        images = jax.vmap(jax.vmap(env_renderer.render_state, (0, None)), (0, None))(states, env_params) # (num_steps, num_eval_levels, ...)
        frames = images.transpose(0, 1, 4, 2, 3) # WandB expects color channel before image dimensions when dealing with animations for some reason
        
        metrics["update_count"] = train_state.update_count
        metrics["eval_returns"] = eval_returns
        metrics["eval_solve_rates"] = eval_solve_rates
        metrics["eval_ep_lengths"]  = episode_lengths
        metrics["eval_animation"] = (frames, episode_lengths)
        metrics["levels"] = jax.vmap(adv_env_renderer.render_level, (0, None))(jax.tree_map(lambda x: x[-1], metrics["levels"]), env_params)
        
        return (rng, train_state), metrics
    
    def eval_checkpoint(og_config):
        """
            This function is what is used to evaluate a saved checkpoint *after* training. It first loads the checkpoint and then runs evaluation.
            It saves the states, cum_rewards and episode_lengths to a .npz file in the `results/run_name/seed` directory.
        """
        rng_init, rng_eval = jax.random.split(jax.random.PRNGKey(10000))
        def load(rng_init, checkpoint_directory: str):
            with open(os.path.join(checkpoint_directory, 'config.json')) as f: config = json.load(f)
            checkpoint_manager = ocp.CheckpointManager(os.path.join(os.getcwd(), checkpoint_directory, 'models'), item_handlers=ocp.StandardCheckpointHandler())

            train_state_og: TrainState = create_train_state(rng_init)
            step = checkpoint_manager.latest_step() if og_config['checkpoint_to_eval'] == -1 else og_config['checkpoint_to_eval']

            loaded_checkpoint = checkpoint_manager.restore(step)
            params = loaded_checkpoint['pro_train_state']['params']
            train_state = train_state_og.replace(pro_train_state=train_state_og.pro_train_state.replace(params=params))
            return train_state.pro_train_state, config
        
        train_state, config = load(rng_init, og_config['checkpoint_directory'])
        states, cum_rewards, episode_lengths = jax.vmap(eval, (0, None))(jax.random.split(rng_eval, og_config["eval_num_attempts"]), train_state)
        save_loc = og_config['checkpoint_directory'].replace('checkpoints', 'results')
        os.makedirs(save_loc, exist_ok=True)
        np.savez_compressed(os.path.join(save_loc, 'results.npz'), states=np.asarray(states), cum_rewards=np.asarray(cum_rewards), episode_lengths=np.asarray(episode_lengths), levels=config['eval_levels'])
        return states, cum_rewards, episode_lengths

    if config['mode'] == 'eval': return eval_checkpoint(config) # evaluate and exit early

    rng = jax.random.PRNGKey(config["seed"])
    rng_init, rng_train = jax.random.split(rng)
    
    train_state = create_train_state(rng_init)
    runner_state = (rng_train, train_state)

    logger.info('Training')
    
    if config["checkpoint_save_interval"] > 0:
        checkpoint_manager = setup_checkpointing(config, train_state, env, env_params)
    start_time = time.time()
    for eval_step in range(config["num_updates"] // config["eval_freq"]):
        runner_state, metrics = train_and_eval_step(runner_state, None)
        curr_time = time.time()
        metrics['time_delta'] = curr_time - start_time
        log_eval(metrics)
        if config["checkpoint_save_interval"] > 0:
            checkpoint_manager.save(eval_step, args=ocp.args.StandardSave(runner_state[1]))
            #checkpoint_manager.wait_until_finished()
    return runner_state[1]

@hydra.main(config_path="config", config_name="main_paired", version_base=None)
def config_main(config: DictConfig):
    if config["num_env_steps"] is not None:
        config["num_updates"] = config["num_env_steps"] // (config["num_train_envs"] * config["num_steps"])

    if config['mode'] == 'eval':
        os.environ['WANDB_MODE'] = 'disabled'
    
    logger.info('Initialising')

    wandb.login()
    main(config, project=config["project"])
    wandb.finish()


if __name__=="__main__":
    config_main()
