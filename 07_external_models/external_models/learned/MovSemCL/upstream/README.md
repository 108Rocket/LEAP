# MovSem: Movement-Semantics Contrastive Learning for Trajectory Similarity - AAAI 2026

## Key Features
- **Movement-Semantics Feature Extraction**: Converts raw GPS data into meaningful movement and semantic features
- **Hierarchical Representation Learning**: Utilizes intra-patch and inter-patch attention to capture multi-level trajectory information
- **Curvature-Guided Augmentation**: A novel data augmentation strategy that improves model robustness
- **Contrastive Learning**: Employs contrastive learning principles to learn discriminative trajectory embeddings

## Project Structure
```
SemMovCL/
├── config.py                    # Configuration settings
├── data/                        # Dataset storage (place datasets here)
├── exp/                         # Experiment logs and snapshots
│   ├── log/
│   └── snapshots/
├── model/                       # Core model definitions
│   ├── hier_sem_encoding.py     # Hierarchical Semantic Encoding layer
│   ├── moco.py                  # MoCo implementation
│   ├── node2vec_.py             # Node2Vec implementation
│   └── semmovcl.py              # Main SemMovCL model
├── task/                        # Specific tasks
│   └── trajsimi.py              # Trajectory similarity task
├── train.py                     # Main training script
├── train_trajsimi.py            # Fine-tuning script for trajectory similarity
└── utils/                       # Utility functions and data handling
    ├── cellspace.py             # Cell space definitions
    ├── data_loader.py           # Data loading utilities
    ├── edwp.py                  # Enhanced Dynamic Time Warping
    ├── preprocessing.py         # Trajectory preprocessing functions
    ├── rdp.py                   # Ramer-Douglas-Peucker algorithm
    ├── tool_funcs.py            # General utility functions
    └── traj.py                  # Trajectory related utilities
```

## Quick Start

### Dataset Download
Extract the corresponding dataset and place it in the `SemMovCL/data/` directory.

### Installation
1. **Setup project**:
   ```bash
   cd SemMovCL
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Training and Testing
#### Pre-training SemMovCL
```bash
python train.py
```

#### Fine-tuning for Trajectory Similarity
To fine-tune the pre-trained SemMovCL to learn to approximate existing heuristic trajectory similarity measures:
```bash
python train_trajsimi.py --trajsimi_measure_fn_name hausdorff
```

### Configuration
You can modify `config.py` to adjust:
- Training parameters (learning rate, batch size, epochs)
- Model configurations (embedding dimensions, attention heads)
- Dataset paths and preprocessing options
- Device settings (CPU/GPU)

## Usage Examples
### Basic Training
```bash
# Train with default settings
python train.py

# Train with custom parameters
python train.py --batch_size 64 --learning_rate 0.001
```

### Fine-tuning for Different Similarity Measures
```bash
# Fine-tune for Hausdorff distance
python train_trajsimi.py --trajsimi_measure_fn_name hausdorff
```

## Acknowledgments
We sincerely thank the authors of TrajCL for generously sharing the codebase and datasets used in this study. Our implementation is built upon the TrajCL framework, which we consider to represent the current SOTA among published methods for trajectory similarity computation. As we do not have the authority to redistribute the datasets, we kindly refer readers to the original TrajCL publication for access and details.
