# main.mojo — Demo runner for exocortex-embed-mojo
#
# Runs example scenarios to demonstrate the embedding operations.
# Not a benchmark — just correctness + usage examples.

from exocortex_embed.vector import (
    dot, norm, cosine_similarity, euclidean_distance,
    add, scale, normalize,
)
from exocortex_embed.matrix import matmul, transpose
from exocortex_embed.random_proj import RandomProjection
from exocortex_embed.index import VectorIndex
from exocortex_embed.quantize import ScalarQuantizer


fn main() raises:
    print("=== exocortex-embed-mojo demo ===")
    print()

    # --- Vector ops ---
    print("--- Vector Operations ---")
    var a = List[Float64](capacity=4)
    a.append(1.0); a.append(2.0); a.append(3.0); a.append(4.0)

    var b = List[Float64](capacity=4)
    b.append(5.0); b.append(6.0); b.append(7.0); b.append(8.0)

    print("dot(a, b) = ", dot(a, b))
    print("norm(a)   = ", norm(a))
    print("cosine(a, b) = ", cosine_similarity(a, b))
    print("euclidean(a, b) = ", euclidean_distance(a, b))

    var summed = add(a, b)
    print("add(a, b) = [", end="")
    for i in range(len(summed)):
        if i > 0:
            print(", ", end="")
        print(summed[i], end="")
    print("]")

    var normalized = normalize(a)
    print("normalize(a) = [", end="")
    for i in range(len(normalized)):
        if i > 0:
            print(", ", end="")
        print(normalized[i], end="")
    print("]")
    print()

    # --- Matrix ops ---
    print("--- Matrix Operations ---")
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
    print("matmul(A, B) =")
    for i in range(len(C)):
        print("  [", end="")
        for j in range(len(C[i])):
            if j > 0:
                print(", ", end="")
            print(C[i][j], end="")
        print("]")

    let AT = transpose(A)
    print("transpose(A) =")
    for i in range(len(AT)):
        print("  [", end="")
        for j in range(len(AT[i])):
            if j > 0:
                print(", ", end="")
            print(AT[i][j], end="")
        print("]")
    print()

    # --- Vector Index ---
    print("--- Vector Index ---")
    var index = VectorIndex(4)

    var v1 = List[Float64](capacity=4)
    v1.append(0.1); v1.append(0.2); v1.append(0.3); v1.append(0.4)
    var v2 = List[Float64](capacity=4)
    v2.append(0.9); v2.append(0.8); v2.append(0.7); v2.append(0.6)
    var v3 = List[Float64](capacity=4)
    v3.append(0.5); v3.append(0.5); v3.append(0.5); v3.append(0.5)

    index.add("doc1", v1)
    index.add("doc2", v2)
    index.add("doc3", v3)

    var query = List[Float64](capacity=4)
    query.append(0.15); query.append(0.25); query.append(0.35); query.append(0.45)

    let results = index.search(query, 3)
    print("Top-3 results for query:")
    for i in range(len(results)):
        print("  ", results[i].get[0](), " score=", results[i].get[1]())
    print()

    # --- Random Projection ---
    print("--- Random Projection ---")
    var rp = RandomProjection(8, 3, seed=12345)
    var high_dim = List[Float64](capacity=8)
    high_dim.append(1.0); high_dim.append(0.0); high_dim.append(1.0); high_dim.append(0.0)
    high_dim.append(1.0); high_dim.append(0.0); high_dim.append(1.0); high_dim.append(0.0)

    let low_dim = rp.project(high_dim)
    print("8D → 3D projection: [", end="")
    for i in range(len(low_dim)):
        if i > 0:
            print(", ", end="")
        print(low_dim[i], end="")
    print("]")
    print()

    # --- Scalar Quantization ---
    print("--- Scalar Quantization ---")
    var quantizer = ScalarQuantizer(4)
    var training = List[List[Float64]](capacity=3)
    training.append(v1)
    training.append(v2)
    training.append(v3)
    quantizer.fit(training)

    let q1 = quantizer.quantize(v1)
    print("quantize(doc1) = [", end="")
    for i in range(len(q1)):
        if i > 0:
            print(", ", end="")
        print(Int(q1[i]), end="")
    print("]")

    let dq1 = quantizer.dequantize(q1)
    print("dequantize     = [", end="")
    for i in range(len(dq1)):
        if i > 0:
            print(", ", end="")
        print(dq1[i], end="")
    print("]")

    print()
    print("=== demo complete ===")
