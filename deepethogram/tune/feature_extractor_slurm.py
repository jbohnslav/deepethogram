import argparse
import logging
import os
import subprocess

import ray

from deepethogram.configuration import make_config
from deepethogram.tune.feature_extractor import tune_feature_extractor

log = logging.getLogger(__name__)
# figure out if we're on slurm or not
try:
    slurm_job_id = os.environ['SLURM_JOB_ID']
    slurm = True
    # hack
    # https://github.com/ray-project/ray/issues/10995
    os.environ["SLURM_JOB_NAME"] = "bash"
except:
    slurm = False


pretrained_dir_remote = '/n/data2/hms/neurobio/harvey/jim/pretrained_models'

project_keys = {
    'woolf': 'woolf_revision_deepethogram',
    'niv':'niv_revision_deepethogram',
    'flies': 'flies_revision_deepethogram',
    'open_field': 'open_field_revision_deepethogram',
    'bohacek_epm': 'bohacek_EPM_deepethogram',
    'bohacek_fst': 'bohacek_FST_deepethogram',
    'bohacek_oft': 'bohacek_OFT_deepethogram', 
    'kc_yd_homecage': 'kc_yd_homecage_deepethogram', 
    'kc_yd_social': 'kc_yd_social_deepethogram'
}

tmp_dir = '/tmp/jpb35'
project_dirs = {key: os.path.join(tmp_dir, project_keys[key]) for key in project_keys.keys()}

# sync FROM the data only directory, only models are flow
remote_dirs_src_base = '/n/data2/hms/neurobio/harvey/jim/deepethogram_revisions_dataonly'
remote_dirs_src = {key: os.path.join(remote_dirs_src_base, project_keys[key]) for key in project_keys.keys()}

remote_dirs_dst_base = '/n/data2/hms/neurobio/harvey/jim/deepethogram_revisions'
remote_dirs_dst = {key: os.path.join(remote_dirs_dst_base, project_keys[key]) for key in project_keys.keys()}

def rsync(src, dst, exclude=None):
    assert os.path.isdir(src)
    assert os.path.isdir(dst)
    command = ['rsync', '-vha', '--no-perms', src, dst]
    if exclude is not None:
        for exc in exclude:
            command += ['--exclude', '{}'.format(exc)]
    print('running rsync with command {}'.format(command))
    subprocess.run(command, check=True)
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run multiple deepethogram models in a row')
    parser.add_argument('-p', '--project', required=True,
                        type=str,
                        help='which project')
    parser.add_argument('--preset',
                        required=True,
                        type=str,
                        help='which model preset, one of deg_f, deg_m, deg_s')
    # parser.add_argument('--model', 
    #                     required=True, 
    #                     type=str, 
    #                     help='one of flow_generator, feature_extractor, or sequence')
    parser.add_argument('-s',
                        '--split',
                        required=True,
                        type=str,
                        help='which split to use. 0-5, except bohacek EPM')
    parser.add_argument('--debug',
                        default='False',
                        type=str,
                        help='whether or not to be in debug mode')
    parser.add_argument('-n',
                        '--notes',
                        type=str,
                        default='',
                        help='notes for this run')
    args = parser.parse_args()

    # make sure bad inputs are handled immediately
    assert args.project in [
        'woolf', 'niv', 'flies', 'open_field', 'bohacek_epm', 'bohacek_fst',
        'bohacek_oft', 'kc_yd_homecage', 'kc_yd_social'
    ]
    assert args.preset in ['deg_f', 'deg_m', 'deg_s']
    split_num = int(args.split)
    assert split_num <= 10
    
    project_dir = project_dirs[args.project]
    # remote_dir = remote_dirs[args.project]
    
    # handle debug
    if args.debug == 'True':
        debug = True
    else:
        debug = False
    log.info('debug mode: {}'.format(debug))

    # set up local and remote dir
    project_dir = project_dirs[args.project]
    remote_dir_src = remote_dirs_src[args.project]
    assert os.path.isdir(remote_dir_src)
    remote_dir_dst = remote_dirs_dst[args.project]
    assert os.path.isdir(remote_dir_dst)

    # handle split names, strings, and files
    split_string = 'split{:02d}'.format(split_num)
    split_dir = os.path.join(project_dir, 'models', split_string)

    # include job ID in notes
    if slurm:
        notes = '{}_{}_{}_split{:02d}'.format(args.notes, slurm_job_id,
                                              args.preset, split_num)
    else:
        notes = '{}_{}_split{:02d}'.format(args.notes, args.preset, split_num)
    if debug:
        notes += '_debug'

    if not os.path.isdir(tmp_dir):
        os.makedirs(tmp_dir)
    
    rsync(remote_dir_src, tmp_dir, exclude=('*.png', '*.ckpt'))
    
    model_path = os.path.join(remote_dir_dst, 'models', split_string)
    assert os.path.isdir(model_path)
    print('model path: {}'.format(model_path))
    assert os.path.isdir(pretrained_dir_remote)
    # USAGE
    # to run locally, type `ray start --head --port 6385`, then run this script
    # asyncio error message: TypeError: __init__() got an unexpected keyword argument 'loop'
    # install aiohttp 3.6.0
    # https://github.com/ray-project/ray/issues/8749
    # for bugs like "could not terminate"
    # "/usr/bin/redis-server 127.0.0.1:6379" "" "" "" "" "" "" ""` due to psutil.AccessDenied (pid=56271, name='redis-server')
    # sudo /etc/init.d/redis-server stop
    # if you have a GPU you can't use for training (e.g. I have a tiny, old GPU just for my monitors) exclude that
    # using command line arguments. e.g. CUDA_VISIBLE_DEVICES=0,1 ray start --head
    
    # ray start --address='10.120.17.247:6379' --redis-password='5241590000000000'
    
    ray.init(address='auto')  #num_gpus=1
    
    config_list = ['config','augs','model/flow_generator','train', 'model/feature_extractor', 
                   'tune/tune', 'tune/feature_extractor']
    run_type = 'train'
    model = 'feature_extractor'
    
    # project_path = projects.get_project_path_from_cl(sys.argv)
    
    cfg = make_config(project_path=project_dir, config_list=config_list, run_type=run_type, model=model, 
                      use_command_line=False, preset=args.preset, debug=debug)
    # make the pretrained path on our persistent location
    cfg.project.pretrained_path = pretrained_dir_remote
    # save our models to persistent location so jobs can be killed without rsyncing
    cfg.project.model_path = model_path
    
    # default settings
    cfg.flow_generator.weights = 'latest'
    cfg.feature_extractor.weights = 'pretrained'
    
    # if debug:
    #     cfg.tune.num_trials=3
    #     cfg.train.steps_per_epoch.train = 100
    #     cfg.train.steps_per_epoch.val = 100
    #     cfg.train.num_epochs = 3
    #     cfg.tune.name = 'tune_feature_extractor_debug'
    # # else:
    cfg.tune.name = 'tune_feature_extractor_{}_narrower'.format(args.preset)
    # cfg.tune.num_trials = 100
    # CUSTOM EDITS HERE
    cfg.compute.batch_size = 32
    cfg.tune.search = 'random'
    cfg.split.reload = True
    cfg.split.file = os.path.join(model_path, split_string + '.yaml')
    assert os.path.isfile(cfg.split.file)
    
    # cfg.tune.grace_period = 5
    # cfg.train.num_epochs = 40
    # cfg.tune.search = 'hyperopt'
    # cfg.tune.name = 'tune_feature_extractor_longhyperopt'
    
    # cfg.tune.name = 'tune_feature_extractor_{}'.format(args.preset)
    
    tune_feature_extractor(cfg) 

