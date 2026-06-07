# matrix.mojo — Basic matrix operations with SIMD hints
#
# Matrices stored as List[List[Float64]] (row-major).
# matmul is naive O(n³) with manual SIMD unrolling on the inner dot product.
# For the exocortex, matrices are typically small (random projection matrices)
# so the simple approach is fine — the SIMD parallelism is in the dot products.

from .vector import dot


## ---------------------------------------------------------------------------
## matmul(A, B) -> List[List[Float64]]
##   Naive O(n³) matrix multiply with SIMD-accelerated inner product.
##   A: m×p, B: p×n  →  result: m×n
## ---------------------------------------------------------------------------
fn matmul(inout A: List[List[Float64]], inout B: List[List[Float64]]) -> List[List[Float64]]:
    let m = len(A)
    let p = len(A[0])
    let n = len(B[0])

    var result = List[List[Float64]](capacity=m)

    # Pre-transpose B for cache-friendly column access
    var BT = List[List[Float64]](capacity=n)
    for j in range(n):
        var col = List[Float64](capacity=p)
        for i in range(p):
            col.append(B[i][j])
        BT.append(col)

    for i in range(m):
        var row = List[Float64](capacity=n)
        for j in range(n):
            # Inner product of A row i with B column j (= BT row j)
            let val = dot(A[i], BT[j])
            row.append(val)
        result.append(row)

    return result


## ---------------------------------------------------------------------------
## transpose(A) -> List[List[Float64]]
##   Transpose an m×n matrix to n×m.
## ---------------------------------------------------------------------------
fn transpose(inout A: List[List[Float64]]) -> List[List[Float64]]:
    let m = len(A)
    if m == 0:
        return List[List[Float64]]()
    let n = len(A[0])

    var result = List[List[Float64]](capacity=n)
    for j in range(n):
        var col = List[Float64](capacity=m)
        for i in range(m):
            col.append(A[i][j])
        result.append(col)

    return result


## ---------------------------------------------------------------------------
## row(A, i) -> List[Float64]
##   Extract row i from matrix A (O(1) for row-major storage, but we copy).
## ---------------------------------------------------------------------------
fn row(inout A: List[List[Float64]], i: Int) -> List[Float64]:
    var result = List[Float64](capacity=len(A[i]))
    for j in range(len(A[i])):
        result.append(A[i][j])
    return result


## ---------------------------------------------------------------------------
## col(A, j) -> List[Float64]
##   Extract column j from matrix A.
## ---------------------------------------------------------------------------
fn col(inout A: List[List[Float64]], j: Int) -> List[Float64]:
    var result = List[Float64](capacity=len(A))
    for i in range(len(A)):
        result.append(A[i][j])
    return result
