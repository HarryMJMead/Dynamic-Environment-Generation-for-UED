"""PACE-inspired scoring using squared vanilla policy-gradient magnitude."""

import jax
import jax.numpy as jnp


def compute_pace_score(train_state, advantages, obs, actions, last_dones, init_hstate):
    """Return one squared policy-gradient norm per rollout level.

    Args:
        train_state: State with ``params`` and an ``apply_fn`` returning
            ``(hidden, policy_distribution, values)`` when called with
            ``(params, (obs, last_dones), init_hstate)``.
        advantages: Raw, unnormalized advantages of shape ``(T, N)``.
        obs: Observation pytree with leaves of shape ``(T, N, ...)``.
        actions: Sampled actions of shape ``(T, N, ...)``. The policy's
            ``log_prob(actions)`` must have shape ``(T, N)``.
        last_dones: Boolean reset flags of shape ``(T, N)``, aligned with
            observations (the previous transition's done, not the current one).
        init_hstate: Initial recurrent-state pytree with leaves ``(N, ...)``.

    Returns:
        Array of shape ``(N,)`` containing ``sum(grad ** 2)`` across all
        parameter leaves, where each level's loss is
        ``-mean(stop_gradient(advantages) * log_prob(actions))``.

    Each column must correspond to one level. All timesteps contribute,
    including incomplete episodes. Initial hidden states are treated as fixed
    rollout data. Shared policy features contribute to the norm; critic-only
    parameters have zero gradients. No optimizer update is performed and the
    train state is unchanged. This is a unit-step vanilla-gradient approximation
    to PACE's parameter-change score, without PPO clipping, entropy or value
    losses. The function can be wrapped in ``jax.jit``.
    """
    advantages = jax.lax.stop_gradient(advantages)
    init_hstate = jax.tree_util.tree_map(jax.lax.stop_gradient, init_hstate)
    # Map levels, retaining the time axis and restoring a singleton batch axis
    # inside each policy call. Reduce each gradient before mapping the next level.
    rollouts = jax.tree_util.tree_map(
        lambda x: jnp.swapaxes(x, 0, 1), (advantages, obs, actions, last_dones)
    )

    def score_level(data):
        rollout, hidden = data
        level_advantages, level_obs, level_actions, level_dones = (
            jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 1), rollout)
        )
        hidden = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 0), hidden)

        def loss_fn(params):
            _, policy, _ = train_state.apply_fn(
                params, (level_obs, level_dones), hidden
            )
            return -jnp.mean(level_advantages * policy.log_prob(level_actions))

        grads = jax.grad(loss_fn)(train_state.params)
        return sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads))

    return jax.lax.map(score_level, (rollouts, init_hstate))
