# random_proj.mojo — Random projection for dimensionality reduction
#
# Implements the Johnson-Lindenstrauss (JL) transform:
#   For n points in ℝ^d, a random projection to k = O(log(n) / ε²) dimensions
#   preserves all pairwise distances within factor (1±ε) with high probability.
#
# References:
#   Johnson & Lindenstrauss, "Extensions of Lipschitz mappings into a Hilbert space"
#   (1984). Indyk & Motwani, "Approximate nearest neighbors: towards removing the
#   curse of dimensionality" (1998).
#
# We use a Gaussian random matrix (each entry ~ N(0, 1/k)) scaled by 1/√k.
# Achlioptas (2001) showed ±1 sparse matrices also work, but Gaussian gives
# tighter JL bounds.

from .matrix import matmul
from .vector import norm


## Simple LCG pseudo-random number generator — zero external deps.
## Constants from Numerical Recipes (Knuth).
struct LCG:
    var state: UInt64

    fn __init__(inout self, seed: Int):
        self.state = UInt64(seed) & 0xFFFFFFFFFFFFFFFF

    fn next(inout self) -> Float64:
        # LCG: state = state * 6364136223846793005 + 1442695040888963407
        self.state = self.state * 6364136223846793005 + 1442695040888963407
        # Map to [0, 1) then shift to (-0.5, 0.5) for approximate Gaussian
        let raw = Float64(self.state >> 33) / Float64(1 << 31)
        return raw - 0.5


## ---------------------------------------------------------------------------
## RandomProjection — JL transform wrapper
## ---------------------------------------------------------------------------
struct RandomProjection:
    var input_dim: Int
    var output_dim: Int
    var projection_matrix: List[List[Float64]]  # output_dim × input_dim
    var _seed: Int

    fn __init__(inout self, input_dim: Int, output_dim: Int, seed: Int = 42):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._seed = seed
        self.projection_matrix = self._init_random_matrix()

    ## Generate the random projection matrix.
    ## Each entry drawn from N(0, 1/output_dim) via LCG + Box-Muller-ish transform.
    ## We use a simple sum-of-uniforms approximation (Central Limit Theorem)
    ## for approximate Gaussian: sum of 12 uniform(0,1) ≈ N(6, 1).
    fn _init_random_matrix(inout self) -> List[List[Float64]]:
        var rng = LCG(self._seed)
        let scale = 1.0 / sqrt(Float64(self.output_dim))

        var matrix = List[List[Float64]](capacity=self.output_dim)
        for i in range(self.output_dim):
            var row = List[Float64](capacity=self.input_dim)
            for j in range(self.input_dim):
                # Approximate Gaussian via CLT: sum 12 uniforms, subtract 6
                var s: Float64 = 0.0
                for _ in range(12):
                    s += rng.next() + 0.5  # shift back to [0,1)
                let gaussian = (s - 6.0) * scale
                row.append(gaussian)
            matrix.append(row)

        return matrix

    ## Project a vector from input_dim → output_dim.
    ## v is treated as a 1 × input_dim "matrix"; result is 1 × output_dim.
    fn project(inout self, inout v: List[Float64]) -> List[Float64]:
        # Wrap v as a 1×input_dim matrix
        var v_matrix = List[List[Float64]](capacity=1)
        var row = List[Float64](capacity=len(v))
        for i in range(len(v)):
            row.append(v[i])
        v_matrix.append(row)

        # Transpose projection_matrix to get input_dim × output_dim
        var PT = List[List[Float64]](capacity=self.input_dim)
        # projection_matrix is output_dim × input_dim
        # We need input_dim × output_dim for matmul(v, PT)
        for j in range(self.input_dim):
            var col = List[Float64](capacity=self.output_dim)
            for i in range(self.output_dim):
                col.append(self.projection_matrix[i][j])
            PT.append(col)

        let result_matrix = matmul(v_matrix, PT)
        return result_matrix[0]


## Helper: compute sqrt (same as in vector.mojo, duplicated to avoid
## cross-module import issues in some Mojo versions).
fn sqrt(x: Float64) -> Float64:
    # Newton's method: 7 iterations for f64 precision
    if x <= 0.0:
        return 0.0
    var s = x
    var i = 0
    while i < 20:
        s = 0.5 * (s + x / s)
        i += 1
    return s
