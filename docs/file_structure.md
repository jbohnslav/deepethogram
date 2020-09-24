# Expected filepaths

To train the DeepEthogram models, we need to be able to find a bunch of files (below). If you use the GUI, this directory
structure will be created for you. 
* models: a list of recent model runs of various types, along with their weights, and their performance
* data
  * for each video, we need the video file itself
  * labels
    * labels are encoded very simply: each column is a behavior, and each row is frame number. Each element of this matrix
    is either {0: no behavior, 1: behavior is present: -1: this frame has not yet been labeled}
  * a file for model outputs
    * for the feature extractor, we save the 512-dimensional image features and 512-dimensional flow features to this file
    * we also save probabilities and predictions (thresholded probabilities) to this file, as well as the thresholds used
  * video statistics: following normal convention in machine learning, we z-score our input data. For images, this is done independently
  for the read, green, and blue channels. We z-score each video as they are added to a project, and save the channel 
  means and std deviations to a file
* project configuration file: holds project-specific information, like behavior names and variables to override. For defaults, see [the default configuration file](../deepethogram/conf/project/project_config.yaml)
  
Therefore, the data loading scripts expect the following consistent folder structure. Note: if you write your own 
dataloaders, you can use whatever file structure you want. 

```bash
project_directory
├── project_config.yaml: See above
├── DATA
|   ├── experiment_0
|   |   ├── experiment_0.avi (or .mp4, etc): the video file
|   |   ├── experiment_0_labels.csv: a label file (see above for formatting)
|   |   ├── experiment_0_outputs.h5: an HDF5 file with extracted features (see above)
|   |   ├── stats.yaml: channel-wise mean and standard deviation
|   |   ├── record.yaml: a yaml file containing the names of all the above files (so other scripts can easily find them, especially if you have multiple video formats in one directory)
|   ├── experiment_1
|   |   ├── experiment_1.avi
|   |   ├── etc...
├── models
|   ├── 200504_flow_generator_None
|   |   ├── checkpoint.pt: the model weights for pytorch
|   |   ├── hydra.yaml: the logs for how hydra built the configuration file
|   |   ├── overrides.yaml: the overrides to the default configuration that the user specified from the command line
|   |   ├── config.yaml: the configuration used to train this model
|   |   ├── split.yaml: the train, validation, test split for this training run
|   |   ├── model_name_definition.pt: the PyTorch model definition
|   |   ├── train.log: log information for this training run
|   |   ├── model_type_metrics.h5: saved metrics for this model. e.g. f1, accuracy, SSIM, depending
|   ├── 200504_feature_extractor_None
|   |   ├── checkpoint.pt: etc...
```