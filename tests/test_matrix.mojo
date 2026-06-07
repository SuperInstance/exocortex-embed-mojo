# test_matrix.mojo — Tests for matrix operations
#
# 8 assertions covering matmul, transpose, row, col.

from exocortex_embed.matrix import matmul, transpose, row, col


fn make_identity(n: Int) -> List[List[Float64]]:
    var M = List[List[Float64]](capacity=n)
    for i in range(n):
        var r = List[Float64](capacity=n)
        for j in range(n):
            if i == j:
                r.append(1.0)
            else:
                r.append(0.0)
        M.append(r)
    return M


fn test_matmul_identity() raises:
    var I = make_identity(3)
    var A = List[List[Float64]](capacity=3)
    for i in range(3):
        var r = List[Float64](capacity=3)
        for j in range(3):
            r.append(Float64(i * 3 + j + 1))
        A.append(r)

    let result = matmul(A, I)
    # A × I = A
    assert(result[0][0] == 1.0, "matmul identity [0][0]")
    assert(result[1][2] == 6.0, "matmul identity [1][2]")
    assert(result[2][2] == 9.0, "matmul identity [2][2]")
    print("  [PASS] matmul identity")


fn test_matmul_known() raises:
    # [[1, 2], [3, 4]] × [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
    var A = List[List[Float64]](capacity=2)
    var r0 = List[Float64](capacity=2)
    r0.append(1.0); r0.append(2.0)
    var r1 = List[Float64](capacity=2)
    r1.append(3.0); r1.append(4.0)
    A.append(r0); A.append(r1)

    var B = List[List[Float64]](capacity=2)
    var c0 = List[Float64](capacity=2)
    c0.append(5.0); c0.append(6.0)
    var c1 = List[Float64](capacity=2)
    c1.append(7.0); c1.append(8.0)
    B.append(c0); B.append(c1)

    let C = matmul(A, B)
    assert(C[0][0] == 19.0, "matmul known [0][0]")
    assert(C[0][1] == 22.0, "matmul known [0][1]")
    assert(C[1][0] == 43.0, "matmul known [1][0]")
    assert(C[1][1] == 50.0, "matmul known [1][1]")
    print("  [PASS] matmul known")


fn test_transpose() raises:
    var A = List[List[Float64]](capacity=2)
    var r0 = List[Float64](capacity=3)
    r0.append(1.0); r0.append(2.0); r0.append(3.0)
    var r1 = List[Float64](capacity=3)
    r1.append(4.0); r1.append(5.0); r1.append(6.0)
    A.append(r0); A.append(r1)

    let AT = transpose(A)
    # 2×3 → 3×2
    assert(len(AT) == 3, "transpose rows")
    assert(len(AT[0]) == 2, "transpose cols")
    assert(AT[0][0] == 1.0, "transpose [0][0]")
    assert(AT[2][1] == 6.0, "transpose [2][1]")
    print("  [PASS] transpose")


fn test_row_col() raises:
    var A = List[List[Float64]](capacity=2)
    var r0 = List[Float64](capacity=3)
    r0.append(1.0); r0.append(2.0); r0.append(3.0)
    var r1 = List[Float64](capacity=3)
    r1.append(4.0); r1.append(5.0); r1.append(6.0)
    A.append(r0); A.append(r1)

    let r = row(A, 1)
    assert(r[0] == 4.0, "row[0]")
    assert(r[2] == 6.0, "row[2]")

    let c = col(A, 1)
    assert(c[0] == 2.0, "col[0]")
    assert(c[1] == 5.0, "col[1]")
    print("  [PASS] row/col")


fn main() raises:
    print("=== matrix tests ===")
    test_matmul_identity()
    test_matmul_known()
    test_transpose()
    test_row_col()
    print("=== all matrix tests passed ===")
