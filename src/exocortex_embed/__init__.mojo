# exocortex-embed-mojo
# Embedding operations for the exocortex — SIMD-accelerated vector math in Mojo
#
# Modules:
#   vector       — SIMD vector ops: dot, norm, cosine, euclidean
#   matrix       — Matrix multiply, transpose
#   random_proj  — Random projection dimensionality reduction
#   index        — Brute-force vector index with top-k search
#   quantize     — Scalar quantization (float64 → int8)

from .vector import (
    dot, norm, cosine_similarity, euclidean_distance,
    add, scale, normalize,
)
from .matrix import matmul, transpose, row, col
from .random_proj import RandomProjection
from .index import VectorIndex
from .quantize import ScalarQuantizer
