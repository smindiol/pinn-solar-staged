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

SWEEP_SIZE = 200
SWEEP_BASE = 42 * 10000


def configuration_seed(index: int) -> int:
    return SWEEP_BASE + index


def configuration(index: int) -> dict:
    """One reproducible configuration of the proposal's objects.

    Not `conftest.py`'s shared `rng` fixture: this is a plain function, never
    a collected pytest item, so it cannot receive a fixture at all -- and it
    needs a generator seeded per `index`, which the fixture's one fixed
    `SEED` does not provide anyway. The two stay separate on purpose.
    """
    rng = random.Random(configuration_seed(index))  # noqa: F841
    raise NotImplementedError


def sweep(size: int = SWEEP_SIZE):
    for index in range(size):
        yield configuration(index)
