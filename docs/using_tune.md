# Hyperparameter tuning

## usage locally
`ray start --head`
`python -m deepethogram.tune.feature_extractor ARGS`

## Usage on slurm cluster
### usage on one node
`ray start --head --num-cpus 16 --num-gpus 2`
in code: `ray.init(address='auto')`

## possible errors
asyncio error message: TypeError: __init__() got an unexpected keyword argument 'loop'
install aiohttp 3.6.0
https://github.com/ray-project/ray/issues/8749
for bugs like "could not terminate"
"/usr/bin/redis-server 127.0.0.1:6379" "" "" "" "" "" "" ""` due to psutil.AccessDenied (pid=56271, name='redis-server')
sudo /etc/init.d/redis-server stop
if you have a GPU you can't use for training (e.g. I have a tiny, old GPU just for my monitors) exclude that
using command line arguments. e.g. CUDA_VISIBLE_DEVICES=0,1 ray start --head