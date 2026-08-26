import numpy as np

from hcrd.persistence import (
    CurvaturePersistenceDiagram,
    PersistenceBar,
    bottleneck_distance,
    curvature_lipschitz_constant,
    curvature_persistence,
    curvature_persistence_distance,
)


def _integrate_curvature(curvature: np.ndarray) -> np.ndarray:
    slopes = np.concatenate(([0.0], np.cumsum(curvature)))
    return np.concatenate(([0.0], np.cumsum(slopes)))


def _diagram(*pairs: tuple[float, float]) -> CurvaturePersistenceDiagram:
    bars = tuple(
        PersistenceBar(birth=birth, death=death, peak_index=index)
        for index, (birth, death) in enumerate(pairs)
    )
    return CurvaturePersistenceDiagram(bars, 0.0, 0)


def test_signed_curvature_diagrams_have_expected_lobes():
    signal = _integrate_curvature(np.asarray([3.0, -1.0, 2.0]))
    signature = curvature_persistence(signal)
    assert signature.positive.essential_birth == 3.0
    assert signature.positive.essential_peak_index == 1
    assert [(bar.birth, bar.death) for bar in signature.positive.bars] == [(2.0, 0.0)]
    assert signature.negative.essential_birth == 1.0
    assert signature.negative.bars == ()


def test_bottleneck_distance_can_prefer_the_diagonal():
    first = _diagram((100.0, 99.0))
    second = _diagram((1.0, 0.0))
    assert bottleneck_distance(first, second) == 0.5
    assert bottleneck_distance(_diagram((2.0, 0.0)), _diagram()) == 1.0


def test_signature_is_affine_invariant_on_an_irregular_grid():
    x = np.asarray([0.0, 0.3, 0.9, 1.8, 3.2, 5.0])
    signal = np.asarray([0.2, -1.0, 0.5, 2.0, 1.5, 4.0])
    first = curvature_persistence(signal, x)
    second = curvature_persistence(signal + 2.5 - 0.7 * x, x)
    assert curvature_persistence_distance(first, second) < 2e-14


def test_randomized_global_stability_bound_uniform_and_irregular():
    rng = np.random.default_rng(918)
    for irregular in (False, True):
        for _ in range(100):
            size = int(rng.integers(6, 35))
            x = (
                np.concatenate(([0.0], np.cumsum(rng.uniform(0.4, 1.6, size - 1))))
                if irregular
                else None
            )
            signal = rng.normal(size=size)
            perturbation = rng.uniform(-1.0, 1.0, size=size) * 10 ** rng.uniform(-7, -2)
            first = curvature_persistence(signal, x)
            second = curvature_persistence(signal + perturbation, x)
            bound = curvature_lipschitz_constant(size, x) * np.max(
                np.abs(perturbation)
            )
            assert curvature_persistence_distance(first, second) <= bound + 2e-12


def test_bars_beyond_twice_the_bound_cannot_all_disappear():
    curvature = np.asarray([4.0, 0.0, 3.0, 0.0, 2.0])
    signal = _integrate_curvature(curvature)
    rng = np.random.default_rng(44)
    perturbation = rng.uniform(-0.01, 0.01, size=signal.size)
    first = curvature_persistence(signal)
    second = curvature_persistence(signal + perturbation)
    delta = 4.0 * np.max(np.abs(perturbation))
    robust_first = sum(bar.lifetime > 2.0 * delta for bar in first.positive.bars)
    assert robust_first == 2
    assert len(second.positive.bars) >= robust_first


def test_two_sample_signal_has_zero_signature_and_constant():
    signature = curvature_persistence([1.0, 3.0])
    assert signature.curvature_constant == 0.0
    assert signature.positive.bars == signature.negative.bars == ()
    assert signature.positive.essential_birth == 0.0
