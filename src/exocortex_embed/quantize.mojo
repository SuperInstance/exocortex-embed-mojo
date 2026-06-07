# quantize.mojo — Scalar quantization for memory-efficient embeddings
#
# Maps Float64 vectors to Int8 vectors for 8× memory reduction (Float64 is 8 bytes,
# Int8 is 1 byte — so 8× not 4×; Float32 → Int8 would be 4×).
#
# Scalar quantization per dimension:
#   For dimension j with range [min_j, max_j]:
#     quantize(x_j) = round(255 × (x_j - min_j) / (max_j - min_j)) - 128
#     → maps to [-128, 127]
#
#   dequantize(q_j) = min_j + (q_j + 128) × (max_j - min_j) / 255
#
#   Reconstruction error per dimension ≈ (max_j - min_j) / 255
#   (uniform quantization noise).
#
# References:
#   Ge et al., "Optimized Product Quantization for Approximate Nearest Neighbor
#   Search" (2013). Scalar quantization is the simplest form; product quantization
#   (PQ) gives better accuracy at the cost of implementation complexity.
#
# Why scalar over product quantization for the exocortex?
#   1. Simpler implementation (this is Mojo, not a FAISS plugin)
#   2. No codebook training needed
#   3. For moderate dimensions (128-768), scalar quantization's error is acceptable
#   4. The quantized_distance is a first-order approximation — good enough for
#      pre-filtering before exact re-ranking


## ---------------------------------------------------------------------------
## ScalarQuantizer — Per-dimension min/max scalar quantization
## ---------------------------------------------------------------------------
struct ScalarQuantizer:
    var dim: Int
    var mins: List[Float64]       # per-dimension minimum
    var maxs: List[Float64]       # per-dimension maximum
    var fitted: Bool

    fn __init__(inout self, dim: Int):
        self.dim = dim
        self.fitted = False
        self.mins = List[Float64](capacity=dim)
        self.maxs = List[Float64](capacity=dim)
        for i in range(dim):
            self.mins.append(0.0)
            self.maxs.append(0.0)

    ## Fit the quantizer to a collection of vectors.
    ## Computes per-dimension min and max across all vectors.
    fn fit(inout self, inout vectors: List[List[Float64]]):
        if len(vectors) == 0:
            return

        # Initialize mins to +inf, maxs to -inf
        for j in range(self.dim):
            self.mins[j] = 1e308
            self.maxs[j] = -1e308

        for i in range(len(vectors)):
            for j in range(self.dim):
                let val = vectors[i][j]
                if val < self.mins[j]:
                    self.mins[j] = val
                if val > self.maxs[j]:
                    self.maxs[j] = val

        self.fitted = True

    ## Quantize a Float64 vector → Int8 vector.
    ## Maps each dimension's value from [min, max] → [-128, 127].
    fn quantize(inout self, inout v: List[Float64]) -> List[Int8]:
        var result = List[Int8](capacity=self.dim)
        for j in range(self.dim):
            let span = self.maxs[j] - self.mins[j]
            if span == 0.0:
                result.append(Int8(0))
            else:
                # Normalize to [0, 255], then shift to [-128, 127]
                let normalized = (v[j] - self.mins[j]) / span * 255.0
                let quantized = Int(normalized) - 128
                # Clamp to [-128, 127]
                let clamped = max(-128, min(127, quantized))
                result.append(Int8(clamped))

        return result

    ## Dequantize an Int8 vector → Float64 vector.
    fn dequantize(inout self, inout q: List[Int8]) -> List[Float64]:
        var result = List[Float64](capacity=self.dim)
        for j in range(self.dim):
            let span = self.maxs[j] - self.mins[j]
            let reconstructed = self.mins[j] + (Float64(Int(q[j])) + 128.0) * span / 255.0
            result.append(reconstructed)

        return result

    ## Approximate Euclidean distance between two quantized vectors.
    ## Computed entirely in Int8 space — no Float64 needed at search time.
    ## Returns the distance as a Float64 for comparison purposes.
    fn quantized_distance(inout self, inout q1: List[Int8], inout q2: List[Int8]) -> Float64:
        var sum_sq: Float64 = 0.0
        for j in range(self.dim):
            let diff = Float64(Int(q1[j]) - Int(q2[j]))
            sum_sq += diff * diff

        # Scale back to original space — each quant step = span/255 per dimension
        # For approximate distance, we can skip per-dim scaling and just return
        # the L2 in quantized space. If you want true distance:
        #   sum_sq += ((diff/255) * (maxs[j] - mins[j]))²
        # But for ranking (relative ordering), unscaled quantized L2 suffices.
        return sum_sq  # unscaled — use for ranking only


## Helper: min for Int
fn min(a: Int, b: Int) -> Int:
    if a < b:
        return a
    return b


## Helper: max for Int
fn max(a: Int, b: Int) -> Int:
    if a > b:
        return a
    return b
