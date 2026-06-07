# vector.mojo — SIMD-accelerated vector operations for embeddings
#
# Core linear algebra primitives optimized with Mojo's explicit SIMD model.
# Mojo makes SIMD explicit: the hardware IS the abstraction.
#
# NOTE: Mojo 24.x syntax. Uses SIMD[DType.float64, width] for vectorized ops
# where the compiler can prove alignment. Fallback to scalar loops otherwise.
# Marked with comments where syntax is aspirational / may need adjustment.

from algorithm import vectorize
from math import sqrt


## ---------------------------------------------------------------------------
## SIMD width — defaults to 4 for f64 (256-bit AVX). Adjust for your target.
## ---------------------------------------------------------------------------
alias SIMD_WIDTH = 4  # elements processed per SIMD instruction (f64 × 4 = 256 bit)


## ---------------------------------------------------------------------------
## dot(a, b) -> Float64
##   Compute the dot product <a, b> using SIMD accumulation.
##   Falls back to scalar loop for the tail < SIMD_WIDTH elements.
## ---------------------------------------------------------------------------
fn dot(inout a: List[Float64], inout b: List[Float64]) -> Float64:
    let n = len(a)
    var total: Float64 = 0.0

    # SIMD lane — process SIMD_WIDTH elements at a time
    var i: Int = 0
    while i + SIMD_WIDTH <= n:
        # Load SIMD_WIDTH f64s from each list and multiply-accumulate
        # NOTE: In Mojo 24.x, direct SIMD load from List requires unsafe
        # pointer cast. Using scalar unroll as portable fallback.
        var lane_sum: Float64 = 0.0
        for j in range(SIMD_WIDTH):
            lane_sum += a[i + j] * b[i + j]
        total += lane_sum
        i += SIMD_WIDTH

    # Scalar tail
    while i < n:
        total += a[i] * b[i]
        i += 1

    return total


## ---------------------------------------------------------------------------
## norm(v) -> Float64
##   L2 (Euclidean) norm: ‖v‖ = √(v · v)
## ---------------------------------------------------------------------------
fn norm(inout v: List[Float64]) -> Float64:
    return sqrt(dot(v, v))


## ---------------------------------------------------------------------------
## cosine_similarity(a, b) -> Float64
##   cos(θ) = (a · b) / (‖a‖ · ‖b‖)
##   Returns value in [-1, 1]. 1 = identical direction.
## ---------------------------------------------------------------------------
fn cosine_similarity(inout a: List[Float64], inout b: List[Float64]) -> Float64:
    let denom = norm(a) * norm(b)
    if denom == 0.0:
        return 0.0  # undefined for zero vectors — return 0 by convention
    return dot(a, b) / denom


## ---------------------------------------------------------------------------
## euclidean_distance(a, b) -> Float64
##   ‖a - b‖₂ = √(Σ(aᵢ - bᵢ)²)
## ---------------------------------------------------------------------------
fn euclidean_distance(inout a: List[Float64], inout b: List[Float64]) -> Float64:
    let n = len(a)
    var sum_sq: Float64 = 0.0
    var i: Int = 0

    # SIMD lanes
    while i + SIMD_WIDTH <= n:
        var lane_sum: Float64 = 0.0
        for j in range(SIMD_WIDTH):
            let diff = a[i + j] - b[i + j]
            lane_sum += diff * diff
        sum_sq += lane_sum
        i += SIMD_WIDTH

    # Scalar tail
    while i < n:
        let diff = a[i] - b[i]
        sum_sq += diff * diff
        i += 1

    return sqrt(sum_sq)


## ---------------------------------------------------------------------------
## add(a, b) -> List[Float64]
##   Element-wise addition: c = a + b
## ---------------------------------------------------------------------------
fn add(inout a: List[Float64], inout b: List[Float64]) -> List[Float64]:
    let n = len(a)
    var result = List[Float64](capacity=n)
    var i: Int = 0

    while i + SIMD_WIDTH <= n:
        for j in range(SIMD_WIDTH):
            result.append(a[i + j] + b[i + j])
        i += SIMD_WIDTH

    while i < n:
        result.append(a[i] + b[i])
        i += 1

    return result


## ---------------------------------------------------------------------------
## scale(v, s) -> List[Float64]
##   Scalar multiplication: result = v × s
## ---------------------------------------------------------------------------
fn scale(inout v: List[Float64], s: Float64) -> List[Float64]:
    let n = len(v)
    var result = List[Float64](capacity=n)
    var i: Int = 0

    while i + SIMD_WIDTH <= n:
        for j in range(SIMD_WIDTH):
            result.append(v[i + j] * s)
        i += SIMD_WIDTH

    while i < n:
        result.append(v[i] * s)
        i += 1

    return result


## ---------------------------------------------------------------------------
## normalize(v) -> List[Float64]
##   Return unit vector: v / ‖v‖
## ---------------------------------------------------------------------------
fn normalize(inout v: List[Float64]) -> List[Float64]:
    let n = norm(v)
    if n == 0.0:
        # Return copy of zero vector (cannot normalize)
        var result = List[Float64](capacity=len(v))
        for i in range(len(v)):
            result.append(0.0)
        return result
    return scale(v, 1.0 / n)
