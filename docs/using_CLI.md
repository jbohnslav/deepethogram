# Using the command line interface

A common way to implement a simple command line interface is to use python's builtin [argparse module](https://docs.python.org/3/library/argparse.html).
However, for this project, we have multiple model types, which share some hyperparameters (such as learning rate) while also 
having unique hyperparameters (such as the loss function used for the optic flow generator). Furthermore, I've put a lot of 
thought into default hyperparameters, so I want to be able to include defaults. Finally, each user must override specific hyperparameters
for their own project, such as the names of the different behaviors. Therefore, for the CLI, we want to be able to 
* have nested arguments, such as train.learning_rate, train.scheduler, etc
* be able to use configuration files to load many parameters at once
* be able to override defaults with our own configuration files
* be able to override everything from the command line

Luckily, [the Hydra python package from Facebook AI](https://hydra.cc/) does all this for us! Therefore, we use hydra for everything.

## Common usage
For all DeepEthogram projects, we [expect a consistent file structure](file_structure.md). Therefore, when using the CLI, always use the flag
`project.config_file=path/to/config/file.yaml`

## examples
To train the flow generator with the larger MotionNet architecture and a batch size of 16: 

`deepethogram.flow_generator.train project.config_file=path/to/config/file.yaml flow_generator.arch=MotionNet compute.batch_size=16`

To train the feature extractor with the ResNet18 base, without the curriculum training, with an initial learning rate of 1e-5: 
`deepethogram.feature_extractor.train project.config_file=path/to/config/file.yaml feature_extractor.arch=resnet18 train.lr=1e-5 feature_extractor.curriculum=false notes=no_curriculum`

To train the flow generator with specific weights loaded from disk, with a specific train/test split, with the DEG_s preset (3D MotionNet): 
`python -m deepethogram.flow_generator.train project.config_file=path/to/config/file.yaml reload.weights=path/to/flow/weights.pt split.file=path/to/split.yaml preset=deg_s`

To train the feature extractor on the secondary GPU with the latest optic flow weights, but a specific feature extractor weights:
`python -m deepethogram.feature_extractor.train project.config_file=path/to/config/file.yaml compute.gpu_id=1 flow_generator.weights=latest feature_extractor.weights=path/to/kinetics_weights.pt`

# Questions?
For any questions on how to use the command line interface for your training, please raise an issue on GitHub. 