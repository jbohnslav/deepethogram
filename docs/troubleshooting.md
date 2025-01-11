# Troubleshooting

Please use the `issues` button on GitHub for any bugs you think you've encountered. During GUI usage, model training,
and model inference, hydra creates a `.log` file (e.g. `train.log`, `main.log`) with various log messages. Please
copy this into your GitHub page. When using the command line, including starting the GUI, use the flag `hydra.verbose=true`.
This will add debugging information to your logs (and also print them to the command line).


# FAQ
#### Model generates poor predictions
The most important factors that determine model performance are as follows:
1. number of data points
2. frequency of behavior (better performance for more common ones)

Please have at least a few hundred frames for each behavior before further inspection is needed.
