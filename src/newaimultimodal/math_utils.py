from __future__ import annotations

"""Small vector helpers used to avoid heavy numeric dependencies."""

import math
import random
from typing import Iterable, List

Vector = List[float]
Matrix = List[List[float]]


def zeros(dim: int) -> Vector:
    """Return a zero-initialised vector."""

    return [0.0 for _ in range(dim)]


def random_matrix(rows: int, cols: int, rng: random.Random) -> Matrix:
    """Sample a dense Gaussian matrix with shape ``(rows, cols)``."""

    return [[rng.gauss(0.0, 1.0) for _ in range(cols)] for _ in range(rows)]


def matvec(matrix: Matrix, vector: Vector) -> Vector:
    """Multiply a matrix with a vector."""

    return [sum(row[j] * vector[j] for j in range(len(vector))) for row in matrix]


def dot(a: Vector, b: Vector) -> float:
    """Compute the scalar product between two vectors."""

    return sum(x * y for x, y in zip(a, b))


def add(a: Vector, b: Vector) -> Vector:
    """Add two vectors component-wise."""

    return [x + y for x, y in zip(a, b)]


def sub(a: Vector, b: Vector) -> Vector:
    """Subtract two vectors component-wise."""

    return [x - y for x, y in zip(a, b)]


def scale(vec: Vector, value: float) -> Vector:
    """Scale a vector by a scalar value."""

    return [x * value for x in vec]


def norm(vec: Vector) -> float:
    """Return the Euclidean norm of ``vec``."""

    return math.sqrt(sum(x * x for x in vec))


def mean(vectors: Iterable[Vector]) -> Vector:
    """Compute the arithmetic mean of a collection of vectors."""

    vectors = list(vectors)
    if not vectors:
        return []
    dim = len(vectors[0])
    accum = zeros(dim)
    for vec in vectors:
        accum = add(accum, vec)
    return [value / len(vectors) for value in accum]


def softmax(values: List[float]) -> List[float]:
    """Stable softmax helper used by the slot attention layer."""

    if not values:
        return []
    max_val = max(values)
    exps = [math.exp(v - max_val) for v in values]
    total = sum(exps)
    if total == 0:
        return [1.0 / len(values) for _ in values]
    return [v / total for v in exps]
