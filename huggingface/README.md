---
language: en
license: other
license_name: deepethogram-academic-noncommercial
license_link: LICENSE
library_name: deepethogram
tags:
- deepethogram
- pytorch
- optical-flow
- animal-behavior
- kinetics
- pretrained
- legacy-archive
---

# DeepEthogram pretrained models — legacy archive

For new projects, use the **[individual model repositories in the DeepEthogram collection](https://huggingface.co/collections/jbohnslav/deepethogram-pretrained-models-6ad61e8af1ac368c56d8deec)**.
Each usable model has its own card, architecture/task tags, checkpoint and checksum manifest.
This repository preserves the historical bundle for compatibility and archival use.

Archived pretrained weights for [DeepEthogram](https://github.com/jbohnslav/deepethogram),
a pipeline for supervised behavior classification from raw video pixels.
These weights initialize optical-flow and feature-extractor training on your own behavior dataset.
They are not ready-to-use classifiers for your behavior labels. No videos, datasets, or trained
sequence classifiers are included.

## Download

Install the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli), then download
into an existing DeepEthogram project's `models` directory:

```bash
hf download jbohnslav/deepethogram-pretrained --revision v1.0 --local-dir /path/to/your_project/models
```

The resulting layout must be `your_project/models/pretrained_models/<run>/...`.
Preserve the run names and nested `flow`/`spatial` directories. Downloading the full repository also
includes this card, the license, and `SHA256SUMS` in the destination directory.
For reproducible downloads, pass `--revision` with a release tag or commit ID.

Verify the model files from that directory:

```bash
cd /path/to/your_project/models
shasum -a 256 -c SHA256SUMS
```

Use the DeepEthogram GUI or training commands to select the pretrained weights. Models retain
the original PyTorch checkpoint format; use DeepEthogram rather than Transformers to load them.

## Contents

The archive contains 30 files (2,058,618,150 bytes). `SHA256SUMS` records the SHA-256 checksum of every archived file. Original checkpoints, YAML hyperparameters, model
definitions, HDF5 metrics, and TIFF training plots are retained without conversion.

| Run | Component | Architecture | Status |
| --- | --- | --- | --- |
| `200221_115158_TinyMotionNet` | Flow generator | TinyMotionNet | Checkpoint present |
| `200310_174416_MotionNet_kinetics` | Flow generator | MotionNet | Checkpoint present |
| `200313_125331_TinyMotionNet3D_kinetics` | Flow generator | TinyMotionNet3D | Checkpoint present |
| `200409_101808_hidden_two_stream_kinetics_degs` | DEG-s feature extractor | ResNet3D-34 | Flow-classifier checkpoint only; incomplete |
| `200410_142156_hidden_two_stream_kinetics_degm` | DEG-m feature extractor | ResNet50 | Full and separate stream checkpoints present |
| `200415_125824_hidden_two_stream_kinetics_degf` | DEG-f feature extractor | ResNet18 | Full and separate stream checkpoints present |

## Training and intended use

The historical configuration files identify Kinetics700 as the pretraining dataset and specify
700 classes for feature extractors. The three flow models provide motion estimates; feature
extractors combine appearance and motion. Fine-tune on your own labeled videos, then train a
sequence model to classify your behaviors. Keep training, validation, and test videos separate.

Historical hyperparameters describe the original training environment and contain old filesystem
paths and YAML serialization tags. They are provenance records, not portable project configs.
The archived `.pt` files use legacy PyTorch serialization.

## Validation

Using PyTorch 2.4.1 and DeepEthogram source revision
`1541eec6958f90673b79c402e10b434c8cb7c868`, all five discoverable models loaded with strict
state-dict matching and produced finite outputs on synthetic CPU inputs. The two feature
extractors returned 700-class outputs. All five nested flow/spatial classifier checkpoints
were readable and contained finite tensors. These checks verify file and architecture
compatibility, not behavior-classification accuracy.

## Known limitations

- The DEG-s folder has `flow/checkpoint.pt` but no top-level checkpoint or spatial checkpoint.
  DeepEthogram's current model discovery skips this incomplete feature extractor. This release
  provides three discoverable flow generators and the DEG-m/DEG-f feature extractors.
  The [individual DEG-s release](https://huggingface.co/jbohnslav/deepethogram-deg-s)
  provides a complete checkpoint.
- DEG-f has no `classification_metrics.h5`; the legacy folders use `hyperparameters.yaml`,
  rather than `config.yaml`. Some inference helpers require newer run files. Use this bundle
  for pretraining initialization and use the new run produced by fine-tuning for inference.
- No task-specific behavior class names, thresholds, or sequence model are supplied.
- The publication's behavior-classification results describe trained pipelines and datasets;
  they are not benchmark scores for this archive alone.

## License and citation

The accompanying [LICENSE](LICENSE) reproduces the DeepEthogram repository's Harvard
Academic and Non-Commercial Research Use Software License and Terms of Use unchanged.
It restricts commercial use and end-user redistribution. This mirror does not grant additional
permissions or relicense the original assets. Commercial licensing inquiries should go to the
Harvard Office of Technology Development as described in that license.

Please cite:

Bohnslav et al. (2021). *DeepEthogram, a machine learning pipeline for supervised behavior
classification from raw pixels*. eLife 10:e63377.
[doi:10.7554/eLife.63377](https://doi.org/10.7554/eLife.63377).
