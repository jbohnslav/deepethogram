# Pretrained model setup

The [DeepEthogram collection](https://huggingface.co/collections/jbohnslav/deepethogram-pretrained-models-6ad61e8af1ac368c56d8deec) groups six individual model repositories.
Download only the models your preset needs. Install DeepEthogram and the
[Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) first.

## Choose a preset

| Preset | Optical-flow generator | Feature extractor |
| --- | --- | --- |
| `deg_f` | [TinyMotionNet](https://huggingface.co/jbohnslav/deepethogram-tinymotionnet) | [DEG-f / ResNet18](https://huggingface.co/jbohnslav/deepethogram-deg-f) |
| `deg_m` | [MotionNet](https://huggingface.co/jbohnslav/deepethogram-motionnet) | [DEG-m / ResNet50](https://huggingface.co/jbohnslav/deepethogram-deg-m) |
| `deg_s` | [TinyMotionNet3D](https://huggingface.co/jbohnslav/deepethogram-tinymotionnet3d) | [DEG-s / ResNet3D-34](https://huggingface.co/jbohnslav/deepethogram-deg-s) |

For the standard DeepEthogram training workflow, download both entries in your preset's row.
Full DEG-f/DEG-m/DEG-s checkpoints include embedded flow weights and can also be loaded directly
without another download; the training workflow selects flow and feature-extractor initialization separately.
These models initialize training on your own labels; a trained behavior pipeline also requires
fine-tuning and a sequence model.

## Individual downloads

Each command creates a separate run folder under your existing project's `models/pretrained_models`.
The `v1.0` release tags pin the pretrained weights.

### TinyMotionNet

[Model card](https://huggingface.co/jbohnslav/deepethogram-tinymotionnet)

```bash
hf download jbohnslav/deepethogram-tinymotionnet --revision v1.0 --local-dir /path/to/project/models/pretrained_models/200221_115158_TinyMotionNet
```

### MotionNet

[Model card](https://huggingface.co/jbohnslav/deepethogram-motionnet)

```bash
hf download jbohnslav/deepethogram-motionnet --revision v1.0 --local-dir /path/to/project/models/pretrained_models/200310_174416_MotionNet_kinetics
```

### TinyMotionNet3D

[Model card](https://huggingface.co/jbohnslav/deepethogram-tinymotionnet3d)

```bash
hf download jbohnslav/deepethogram-tinymotionnet3d --revision v1.0 --local-dir /path/to/project/models/pretrained_models/200313_125331_TinyMotionNet3D_kinetics
```

### DEG-f

[Model card](https://huggingface.co/jbohnslav/deepethogram-deg-f)

```bash
hf download jbohnslav/deepethogram-deg-f --revision v1.0 --local-dir /path/to/project/models/pretrained_models/200415_125824_hidden_two_stream_kinetics_degf
```

### DEG-m

[Model card](https://huggingface.co/jbohnslav/deepethogram-deg-m)

```bash
hf download jbohnslav/deepethogram-deg-m --revision v1.0 --local-dir /path/to/project/models/pretrained_models/200410_142156_hidden_two_stream_kinetics_degm
```

### DEG-s

[Model card](https://huggingface.co/jbohnslav/deepethogram-deg-s)

```bash
hf download jbohnslav/deepethogram-deg-s --revision v1.0 --local-dir /path/to/project/models/pretrained_models/200409_101808_hidden_two_stream_kinetics_degs
```

## Verify and use

From each downloaded run folder:

```bash
shasum -a 256 -c SHA256SUMS
```

DeepEthogram discovers each `checkpoint.pt` using the neighboring `hyperparameters.yaml`.
Keep the run folder names shown above. Select the appropriate preset in the GUI or CLI;
see [CLI training](using_CLI.md) for project setup and training commands.
Each model card also provides a strict-load Python example and synthetic forward-pass example.
No new package dependency or model-loader changes are required.

The historical YAML files contain original filesystem paths; they describe pretraining, rather
than serving as a portable project config. Use the new run created by fine-tuning for behavior
inference: the initialization repos do not contain the newer `config.yaml` and inference metrics.

## Legacy archive

The [original six-folder bundle](https://huggingface.co/jbohnslav/deepethogram-pretrained)
remains available with historical plots, metrics, separate stream snapshots and an older incomplete DEG-s folder.
The individual DEG-s repository provides the complete checkpoint used in the test suite.
Its structure differs: downloading that bundle into `models` creates the entire `pretrained_models` tree.
The [historical Google Drive ZIP](https://drive.google.com/file/d/1ntIZVbOG1UAiFVlsAAuKEBEVCVevyets/view)
is retained as a fallback; it was not used for these releases.
