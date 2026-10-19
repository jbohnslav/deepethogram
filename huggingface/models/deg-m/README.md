---
model_name: DeepEthogram DEG-m
language: en
license: other
license_name: deepethogram-academic-noncommercial
license_link: LICENSE
library_name: deepethogram
pipeline_tag: video-classification
tags:
- deepethogram
- pytorch
- video-classification
- deg-m
- resnet50
- kinetics700
- animal-behavior
- pretrained
---

# DeepEthogram DEG-m

DEG-m is a Kinetics700-pretrained hidden two-stream feature extractor for [DeepEthogram](https://github.com/jbohnslav/deepethogram).
This repository contains this model's checkpoint and historical training hyperparameters.
See the [DeepEthogram collection](https://huggingface.co/collections/jbohnslav/deepethogram-pretrained-models-6ad61e8af1ac368c56d8deec) for the other models.

## Model details

| Property | Value |
| --- | --- |
| Architecture | resnet50 with MotionNet and average two-stream fusion |
| Task | video-classification |
| Input | 11 RGB frames stacked as `(N, 33, H, W)` |
| Output | 700 logits per input clip, before any task-specific behavior fine-tuning |
| Pretraining data | Kinetics700, as recorded in the historical hyperparameters |
| Framework | PyTorch; DeepEthogram |

The full checkpoint includes its own MotionNet optical-flow generator, RGB and flow
classifiers, and fusion parameters. Loading this model directly needs no other model download.
For DeepEthogram's normal training workflow, download the separate
[MotionNet](https://huggingface.co/jbohnslav/deepethogram-motionnet) initialization checkpoint too:
the training code initializes flow and feature-extractor components separately.

## Download and load

Install DeepEthogram and the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli).
Download only this model into an existing DeepEthogram project's pretrained directory:

```bash
hf download jbohnslav/deepethogram-deg-m --revision v1.0 --local-dir /path/to/project/models/pretrained_models/200410_142156_hidden_two_stream_kinetics_degm
```

From that run directory, verify the artifacts with `shasum -a 256 -c SHA256SUMS`.
The `checkpoint.pt` and `hyperparameters.yaml` at its root let DeepEthogram discover
this model using its existing project layout.

To load directly in Python, set `run_dir` to the directory downloaded above:

```python
from pathlib import Path
import torch
from deepethogram import utils

from deepethogram.feature_extractor.models.hidden_two_stream import deg_m
model = deg_m(num_classes=700, reload_imagenet=False)
run_dir = Path('/path/to/project/models/pretrained_models/200410_142156_hidden_two_stream_kinetics_degm')
state_dict, _, _ = utils.load_state_dict_from_file(run_dir / 'checkpoint.pt')
model.load_state_dict(state_dict, strict=True)
model.eval()
with torch.inference_mode():
    output = model(torch.zeros(1, 33, 64, 64))
```

The zero-valued example checks shapes and loading. For real videos use DeepEthogram's
video preprocessing and normalization. Historical settings are in `hyperparameters.yaml`;
`model_config.json` summarizes architecture and input/output conventions.

## Intended use and limitations

Use these weights to initialize feature-extractor training on your own labeled behavior videos,
then train a sequence model. The 700-class pretraining head is not a classifier for your animal
behavior labels. Kinetics class-index names and trained behavior thresholds are not supplied.

Historical YAML files contain old filesystem paths and legacy serialization tags; they are
provenance records, not portable project configs. Checkpoints retain their original PyTorch
serialization. Use DeepEthogram, rather than Transformers `AutoModel`, to load them.

This release contains initialization weights, not a complete trained behavior inference run:
newer inference helpers expect `config.yaml` and associated metrics from your fine-tuned run.
The complete historical plots, metrics, and additional stream snapshots remain in the
[legacy archive](https://huggingface.co/jbohnslav/deepethogram-pretrained).

## Validation

`SHA256SUMS` covers the checkpoint, training hyperparameters, license and model configuration.
Validated with PyTorch 2.4.1 and DeepEthogram source revision
`1541eec6958f90673b79c402e10b434c8cb7c868`: strict parameter loading and finite CPU
outputs on the synthetic input above. This is a compatibility check, not an accuracy benchmark.

## License and citation

[LICENSE](LICENSE) reproduces the original DeepEthogram Harvard Academic and Non-Commercial
Research Use Software License and Terms of Use unchanged. It restricts commercial use and
end-user redistribution. This mirror grants no additional permissions and does not relicense
the assets; commercial inquiries should go to Harvard OTD as described in the license.

Bohnslav et al. (2021). *DeepEthogram, a machine learning pipeline for supervised behavior
classification from raw pixels*. eLife 10:e63377.
[doi:10.7554/eLife.63377](https://doi.org/10.7554/eLife.63377).
