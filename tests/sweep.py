"""Randomized configuration sweep shared by the audit and the remedy checks.

One fixed size for every claim: a property surviving SWEEP_SIZE independent
configurations belongs to the formulation, not to a seed.

SWEEP_BASE offsets the block by the run's seed so separate runs audit disjoint
configurations. Without it, agreement between runs would confirm determinism
and nothing else.

`configuration` builds one instance of whatever the proposal's objects are.
Vary what the proposal allows to vary — cardinalities, dispersion, shifts — so
a claim that only holds for one shape shows up as a rate below the full sweep.

Backend-neutral on purpose: no array library is imported here, so this file
never decides the implementation's backend (`__backend__` in
`src/<Package>/__init__.py` does). `configuration_seed(index)` is the seed of
one configuration; build the backend's own generator from it, e.g.
`numpy.random.default_rng(configuration_seed(index))` or
`torch.Generator().manual_seed(configuration_seed(index))`.
"""

from __future__ import annotations

import random
from functools import lru_cache

SWEEP_SIZE = 200
SWEEP_BASE = 42 * 10000


def configuration_seed(index: int) -> int:
    return SWEEP_BASE + index


@lru_cache(maxsize=1)
def _train_draw() -> tuple:
    """The sweep's 200 distinct training devices: one seeded draw without replacement from the
    device-level training split (seed 42), so no two configurations share a device and the
    validation and test devices are never touched. Imported lazily to keep this module free of
    array libraries."""
    from Staged_PINN_Solar import data

    train = data.split_devices(data.inventory(), seed=data.SEED).train
    order = random.Random(SWEEP_BASE).sample(range(len(train)), SWEEP_SIZE)
    return tuple(train[i] for i in order)


def configuration(index: int) -> dict:
    """One reproducible configuration: a distinct training device of the dPV E1A+E1B set.

    The proposal's varying object is the device (its doping, diffusion lengths and, in E1B, the
    layer thicknesses), and every property below is measured on the stored reference solution
    of that device over all its voltages and nodes.
    """
    return {"index": index, "seed": configuration_seed(index), "ref": _train_draw()[index]}


def sweep(size: int = SWEEP_SIZE):
    for index in range(size):
        yield configuration(index)
