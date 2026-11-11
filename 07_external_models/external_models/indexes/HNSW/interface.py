"""FAISS HNSW boundary used by the ConvTraj-vector control."""


def build(vectors, *, m=32, ef_construction=200):
    import faiss
    import numpy as np
    values = np.ascontiguousarray(vectors, dtype=np.float32)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Expected finite database vectors [N,dimension]')
    if m < 1 or ef_construction < 1:
        raise ValueError('Construction parameters must be positive')
    faiss.omp_set_num_threads(1)
    index = faiss.IndexHNSWFlat(values.shape[1], m, faiss.METRIC_L2)
    index.hnsw.efConstruction = ef_construction
    index.add(values)
    return index


def search(index, queries, *, top_k=20, ef=80, threads=4):
    import faiss
    import numpy as np
    values = np.ascontiguousarray(queries, dtype=np.float32)
    if (values.ndim != 2 or values.shape[1] != index.d or not np.isfinite(values).all()
            or not 1 <= top_k <= index.ntotal or ef < top_k or threads < 1):
        raise ValueError('Invalid query shape, k, ef or thread count')
    faiss.omp_set_num_threads(threads)
    index.hnsw.efSearch = ef
    return index.search(values, top_k)
