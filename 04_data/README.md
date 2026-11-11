# Datasets

[Preparation tools](../09_tools/README.md) / [Reproduction guide](../11_docs/reproduction.md)

Dataset directories contain README files and empty `.gitkeep` placeholders.
Full datasets and generated geometry/reference arrays are not bundled.
Published checkpoints and default reproduction jobs cover Porto only; the
other dataset entries below document the paper's data sources.

| Dataset | Geometry folder | Hausdorff target |
| --- | --- | --- |
| [100K Porto](100k/README.md) | `100k/porto/` | Vertex to segment |
| [100K Germany](100k/README.md) | `100k/germany/` | Vertex to segment |
| [SDD Porto](sdd/README.md) | `sdd/porto/` | Point set |
| [SDD Geolife](sdd/README.md) | `sdd/geolife/` | Point set |
| [SDD Chengdu](sdd/README.md) | `sdd/chengdu/` | Point set |

The [100K guide](100k/README.md) links the original sources, processed inputs,
and selection code. The [SDD guide](sdd/README.md) links the published benchmark,
its split preparation, and distance checks. [Preparation code](../09_tools/data_preparation/README.md)
distinguishes historical generation scripts from the runnable format converter.
Retain dataset licenses, source record IDs, query membership, and ordering.

## Packed Geometry

Each folder needs `database.npz`. SDD additionally needs `queries.npz`.
100K uses its database as queries and excludes the self match. Each NPZ contains:

| Key | Shape and type |
| --- | --- |
| `points` | `[P, 2]` original ordered coordinates, preferably float64 |
| `offsets` | `[N+1]` integer offsets, starting at zero and ending at P |
| `geometry_mean` | `[2]`, exactly the chosen checkpoint's normalization |
| `geometry_std` | `[2]`, exactly the chosen checkpoint's positive scales |
| `source_ids` (optional) | `[N]` integer source record IDs, retained when supplied |

Every trajectory has at least two finite original vertices. Keep all vertices,
directions, trajectory order and coordinate units. The 100K checkpoint convention
uses projected meter coordinates, while the SDD checkpoints use their recorded
SDD coordinates. The packer performs no projection, resampling or normalization.

[pack_data.py](../09_tools/pack_data.py) accepts packed NPZ or a JSON list of trajectories and writes
the above schema using the selected checkpoint's normalization. NPZ inputs may use
either `offsets` or per-trajectory `lengths`, as emitted by `prepare_100k.py`.
If both are supplied, they must describe the same trajectories. Source IDs are
preserved in trajectory order and are not replaced by packed row indices.

```bash
python 09_tools/pack_data.py --input prepared_porto.json \
  --checkpoint sdd.porto.hausdorff --output 04_data/sdd/porto/database.npz
```

This utility is a file-format converter. It does not recreate the paper's split
membership, infer coordinate units or establish that arbitrary data matches a
published benchmark. Select the intended original records before packing.

## References and Training Inputs

Place FP64 Top-ID references in each `references/` folder. Reference rows must
match the packed geometry order. Record SHA256 for the geometry files in the
run configuration, as shown in the [reproduction guide](../11_docs/reproduction.md).

The `train/` and `validation/` folders reserve paths for independently prepared
training inputs and original-distance labels. They must not contain evaluation
answers used to select checkpoints. The paper fixes hyperparameters on Porto
and trains city-specific parameters.

Run `python -m leap check --require-data` after placing the files. Blank
placeholders are never treated as valid datasets or evaluated as empty data.
