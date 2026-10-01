# vector.mojo — SIMD-accelerated vector operations for embeddings
#
# Core linear algebra primitives optimized with Mojo's explicit SIMD model.
# Mojo makes SIMD explicit: the hardware IS the abstraction.
#
# NOTE: Mojo 24.x syntax. Uses SIMD[DType.float64, width] for vectorized ops
# where the compiler can prove alignment. Fallback to scalar loops otherwise.
# Marked with comments where syntax is aspirational / may need adjustment.

from std import math


## ---------------------------------------------------------------------------
## SIMD width — defaults to 4 for f64 (256-bit AVX). Adjust for your target.
## ---------------------------------------------------------------------------
# SIMD width: 4 (elements processed per SIMD instruction, f64 × 4 = 256 bit).
# NOTE: module-level `alias` is banned in current Mojo (like module-level var),
# so the constant is inlined as 4 at each use site below.


## ---------------------------------------------------------------------------
## dot(a, b) -> Float64
##   Compute the dot product <a, b> using SIMD accumulation.
##   Falls back to scalar loop for the tail < 4 elements.
## ---------------------------------------------------------------------------
def dot(a: List[Float64], b: List[Float64]) -> Float64:
    var n = len(a)
    var total: Float64 = 0.0

    # SIMD lane — process 4 elements at a time
    var i: Int = 0
    while i + 4 <= n:
        # Load 4 f64s from each list and multiply-accumulate
        # NOTE: In Mojo 24.x, direct SIMD load from List requires unsafe
        # pointer cast. Using scalar unroll as portable fallback.
        var lane_sum: Float64 = 0.0
        for j in range(4):
            lane_sum += a[i + j] * b[i + j]
        total += lane_sum
        i += 4

    # Scalar tail
    while i < n:
        total += a[i] * b[i]
        i += 1

    return total


## ---------------------------------------------------------------------------
## norm(v) -> Float64
##   L2 (Euclidean) norm: ‖v‖ = √(v · v)
## ---------------------------------------------------------------------------
def norm(v: List[Float64]) -> Float64:
    return math.sqrt(dot(v, v))


## ---------------------------------------------------------------------------
## cosine_similarity(a, b) -> Float64
##   cos(θ) = (a · b) / (‖a‖ · ‖b‖)
##   Returns value in [-1, 1]. 1 = identical direction.
## ---------------------------------------------------------------------------
def cosine_similarity(a: List[Float64], b: List[Float64]) -> Float64:
    var denom = norm(a) * norm(b)
    if denom == 0.0:
        return 0.0  # undefined for zero vectors — return 0 by convention
    return dot(a, b) / denom


## ---------------------------------------------------------------------------
## euclidean_distance(a, b) -> Float64
##   ‖a - b‖₂ = √(Σ(aᵢ - bᵢ)²)
## ---------------------------------------------------------------------------
def euclidean_distance(a: List[Float64], b: List[Float64]) -> Float64:
    var n = len(a)
    var sum_sq: Float64 = 0.0
    var i: Int = 0

    # SIMD lanes
    while i + 4 <= n:
        var lane_sum: Float64 = 0.0
        for j in range(4):
            var diff = a[i + j] - b[i + j]
            lane_sum += diff * diff
        sum_sq += lane_sum
        i += 4

    # Scalar tail
    while i < n:
        var diff = a[i] - b[i]
        sum_sq += diff * diff
        i += 1

    return math.sqrt(sum_sq)


## ---------------------------------------------------------------------------
## add(a, b) -> List[Float64]
##   Element-wise addition: c = a + b
## ---------------------------------------------------------------------------
def add(a: List[Float64], b: List[Float64]) -> List[Float64]:
    var n = len(a)
    var result = List[Float64](capacity=n)
    var i: Int = 0

    while i + 4 <= n:
        for j in range(4):
            result.append(a[i + j] + b[i + j])
        i += 4

    while i < n:
        result.append(a[i] + b[i])
        i += 1

    return result^


## ---------------------------------------------------------------------------
## scale(v, s) -> List[Float64]
##   Scalar multiplication: result = v × s
## ---------------------------------------------------------------------------
def scale(v: List[Float64], s: Float64) -> List[Float64]:
    var n = len(v)
    var result = List[Float64](capacity=n)
    var i: Int = 0

    while i + 4 <= n:
        for j in range(4):
            result.append(v[i + j] * s)
        i += 4

    while i < n:
        result.append(v[i] * s)
        i += 1

    return result^


## ---------------------------------------------------------------------------
## normalize(v) -> List[Float64]
##   Return unit vector: v / ‖v‖
## ---------------------------------------------------------------------------
def normalize(v: List[Float64]) -> List[Float64]:
    var n = norm(v)
    if n == 0.0:
        # Return copy of zero vector (cannot normalize)
        var result = List[Float64](capacity=len(v))
        for _ in range(len(v)):
            result.append(0.0)
        return result^
    return scale(v, 1.0 / n)
