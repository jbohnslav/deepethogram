# Maintaining the DeepEthogram model releases

Primary releases are six individual model repos grouped in the
[DeepEthogram collection](https://huggingface.co/collections/jbohnslav/deepethogram-pretrained-models-6ad61e8af1ac368c56d8deec).
`models.json` records repo IDs, original run names, exact commits and verification evidence.
`models/<slug>/` contains the corresponding model card, machine-readable configuration and checksum manifest.
Model binaries remain outside Git.

## Per-model layout

Stage only the chosen run's original `checkpoint.pt` and `hyperparameters.yaml`, this repository's
unchanged `license.txt` as `LICENSE`, and that model's card, `model_config.json` and `SHA256SUMS`:

```text
README.md
LICENSE
checkpoint.pt
hyperparameters.yaml
model_config.json
SHA256SUMS
```

Keep checkpoint bytes unchanged. The configuration summary is specific to DeepEthogram;
it is not a Transformers configuration. Model cards specify the library, architecture and task.
DEG-f/DEG-m/DEG-s use the supported `video-classification` pipeline tag. Flow generators use the custom
`optical-flow` tag, since Hugging Face's current task registry has no optical-flow pipeline category.

Each manifest covers weights, hyperparameters, license and model configuration. Recompute the
manifest whenever those files change. Verify the staged files using `shasum -a 256 -c SHA256SUMS`.
Upload a dedicated staging directory, not the source checkout:

```bash
hf upload jbohnslav/deepethogram-MODEL /path/to/staged-model . --type model
```

Download the uploaded commit into a fresh run folder, verify checksums, execute the model card's
Python example, and check project discovery from its parent pretrained_models directory. Check
strict state-dict loading and a finite-output CPU forward pass with the corresponding architecture.
Read back Hub metadata to verify the model's task/library tags. Only then tag the verified commit.

The initial per-model release tag is `v1.0`. Model cards and user documentation pin it.
When updating weights, create a new tag rather than moving an existing release tag.
Card-only fixes (`README.md`, which is not in `SHA256SUMS`) can go straight to `main`. They don't need a new tag, and `models.json` keeps pinning the verified release commit.

## Collection and legacy archive

Add each verified model to the collection. Document the training pairs: TinyMotionNet + DEG-f,
MotionNet + DEG-m, and TinyMotionNet3D + DEG-s. Their full feature-extractor checkpoints include embedded flow weights
for direct Python loading, while the normal training workflow loads the components separately.

The older [deepethogram-pretrained](https://huggingface.co/jbohnslav/deepethogram-pretrained)
repo remains an explicitly labeled legacy archive. Its card is `README.md` in this directory;
its original 30-file checksum manifest is `SHA256SUMS`. It retains plots, metrics, model definitions
and stream checkpoints, including an older incomplete DEG-s folder. The individual DEG-s
release provides the complete checkpoint from the existing test fixture archive.

## CI checkpoint contract

`setup_tests.py` reads `models.json`. For every model it checks that the release tag resolves to the pinned
`commit`, downloads that commit into `tests/DATA/huggingface_checkpoints/`, and verifies all files against the
checked-in per-model checksum manifest. It then rebuilds the archive's `pretrained_models` directory with only
these runs, so stale folders cannot shadow the pinned weights. Existing fixture folders still run this step.
CPU CI caches the download directory, keyed on `models.json`; the Modal GPU runner copies only its video
fixtures from the persistent test-data volume and downloads the checkpoints fresh.
`tests/test_pretrained_models.py` checks artifact identity, strict architecture matching, finite forward outputs
and discovery for all models. `tests/test_setup_tests.py` checks that `models.json` agrees with each
`model_config.json` and that the Docker test images ship `huggingface/`.
When releasing new weights, update commits/checksums together and run setup plus the CPU suite.
