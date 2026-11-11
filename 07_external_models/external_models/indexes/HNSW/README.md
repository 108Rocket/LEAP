# HNSW over ConvTraj Vectors

This is the paper's extra vector-index control, not another trajectory encoder.
The control uses the original [FAISS](https://github.com/facebookresearch/faiss)
`IndexHNSWFlat` with squared L2 over ConvTraj vectors. It does not use hnswlib.
FAISS is an external runtime dependency; its full library source is not bundled.

[interface.py](interface.py) only calls the native FAISS index. The recorded
control uses `M=32`, `efConstruction=200`, one construction thread, four query
threads and `efSearch` in `{20,40,80,160,320,640}`. Query encoding is separate
from vector search. Three-seed LEAP timing rules do not redefine this separate
single-checkpoint control.

Supply your own ConvTraj vectors, model checkpoint and FAISS installation.
The interface does not retrain or modify ConvTraj and does not implement HNSW.

[All external methods](../../README.md) | [LEAP overview](../../../../README.md)
