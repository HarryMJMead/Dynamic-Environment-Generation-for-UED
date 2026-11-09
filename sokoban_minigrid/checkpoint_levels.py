"""Uniformly sample a fixed pool of levels from ACCEL/PLR checkpoints."""
from dataclasses import fields
import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from jaxued.environments.maze import ObservedLevel

logger = logging.getLogger(__name__)


def _load_checkpoint_levels(directory, step):
    """Load the populated portion of one buffer, without policy or optimizer state."""
    template = {
        "sampler": {
            "size": 0,
            "levels": {field.name: 0 for field in fields(ObservedLevel)},
        }
    }
    # Host arrays let us load GPU checkpoints without their original devices.
    restore_args = jax.tree_util.tree_map(
        lambda _: ocp.RestoreArgs(restore_type=np.ndarray), template
    )
    manager = ocp.CheckpointManager(
        str(directory / "models"),
        item_handlers=ocp.PyTreeCheckpointHandler(),
        options=ocp.CheckpointManagerOptions(create=False),
    )
    try:
        if step == -1:
            step = manager.latest_step()
        checkpoint = manager.restore(
            step,
            args=ocp.args.PyTreeRestore(
                item=template, restore_args=restore_args, transforms={},
            ),
        )
    finally:
        manager.close()

    buffer = checkpoint["sampler"]
    size = int(buffer["size"])
    levels = {name: values[:size] for name, values in buffer["levels"].items()}
    source_info = {"directory": str(directory), "step": int(step), "num_levels": size}
    logger.info("Loaded %s step %s: %d levels", directory, step, size)
    return levels, source_info


def make_checkpoint_level_sampler(sources, base_directory):
    """Return a callable ``rng -> ObservedLevel`` and resolved source information.

    Sources specify run directories and optional steps (-1 or omitted means latest).
    Sampling is uniform over all populated entries, with duplicates retained.
    """
    level_batches = []
    resolved_sources = []
    for source in sources:
        directory = Path(base_directory) / Path(source["directory"]).expanduser()
        levels, source_info = _load_checkpoint_levels(
            directory.resolve(), source.get("step", -1)
        )
        level_batches.append(levels)
        resolved_sources.append(source_info)

    # Concatenate each field on the host before moving the combined pool to JAX.
    pool = ObservedLevel(**{
        name: jnp.asarray(np.concatenate([batch[name] for batch in level_batches]))
        for name in level_batches[0]
    })
    num_levels = pool.wall_map.shape[0]
    logger.info("Checkpoint level pool: %d entries from %d sources", num_levels, len(sources))

    def sample_level(rng):
        index = jax.random.randint(rng, (), 0, num_levels)
        return jax.tree_util.tree_map(lambda values: values[index], pool)

    return sample_level, resolved_sources
