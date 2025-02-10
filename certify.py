import argparse
import logging
from omegaconf import OmegaConf
import os
from time import time
import datetime
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from core import Smooth
from datasets import get_dataset, DATASETS, get_num_classes
import utils
from runners.diffpure_guided_densepure import GuidedDiffusion
import timm
from networks import *
import getpass


class DensePure_Certify(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config

        self.classifier = timm.create_model('beit_large_patch16_512', pretrained=True).cuda()
        self.classifier.eval()
        self.runner = GuidedDiffusion(args, config)

        # self.register_buffer('counter', torch.zeros(1, device=config.device))
        self.tag = None

    def reset_counter(self):
        self.counter = torch.zeros(1, dtype=torch.int)

    def set_tag(self, tag=None):
        self.tag = tag

    def forward(self, x, sample_id, original_x):
        # print(sample_id)
        # print(x)
        # print(x.shape)
        # print(x.min().item(), x.max().item())
        # print((2*x-1).min().item(), (2*x-1).max().item())
        counter = self.counter.item()
        # if counter % 5 == 0:
        #     print(f'diffusion times: {counter}')

        # print(original_x.shape)
        # print(original_x.min(), original_x.max())

        start_time = time()
        x_re = self.runner.image_editing_sample((x - 0.5) * 2, original_x, bs_id=counter,  tag=self.tag)
        # print(x_re.min(), x_re.max())

        minutes, seconds = divmod(time() - start_time, 60)

        # if self.args.save_info:
        #     np.save(self.args.image_folder+'/'+str(sample_id)+'-'+str(counter)+'-img_after_purify.npy',x_re.clone().detach().cpu().numpy())

        x_re = F.interpolate(x_re, size=(512, 512), mode='bicubic')
        with torch.no_grad():
            self.classifier.eval()
            out = self.classifier(x_re)

        # if self.args.save_info:
        #     np.save(self.args.image_folder+'/'+str(sample_id)+'-'+str(counter)+'-logits.npy',out.clone().detach().cpu().numpy())

        self.counter += 1

        return out


def purified_certify(model, dataset, args, config):
    # ---------------- evaluate certified robustness of diffpure + classifier ----------------
    ngpus = torch.cuda.device_count()
    model_ = model
    if ngpus > 1:
        model_ = model.module

    model_.reset_counter()
    smoothed_classifier_diffpure = Smooth(model, get_num_classes(config.dataset.domain), config.certify.sigma)
    file_path = os.path.join(config.log_dir, 'certify')
    f = open(file_path, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    for i in range(0, 49001, 1000):
        (x, label) = dataset[i]

        # print(f"shape is {x.shape}")


        before_time = time()
        # certify the prediction of g around x
        x = x.cuda()
        label = torch.tensor(label,dtype=torch.int).cuda()
        prediction, radius, n0_predictions, n_predictions = smoothed_classifier_diffpure.certify(x, config.certify.N0, config.certify.N, i, config.certify.alpha, config.certify.batch_size, clustering_method=None)
        after_time = time()
        correct = int(prediction == label)
        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)
        # if args.save_predictions:
        #     np.save(args.predictions_path+str(i)+'-'+str(args.reverse_seed)+'-n0_predictions.npy',n0_predictions)
        #     np.save(args.predictions_path+str(i)+'-'+str(args.reverse_seed)+'-n_predictions.npy',n_predictions)
    f.close()


def robustness_eval(args, config, device):
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    config.log_dir = os.path.join("logs", config.guide_type, config.scaling_type, "scale"+str(config.scale), now)
    if args.toolkit : config.log_dir = os.path.join('/mnt/home/DensePure', config.log_dir)
    os.makedirs(config.log_dir, exist_ok=True)
    OmegaConf.save(config, os.path.join(config.log_dir, 'config.yaml'))


    logger = utils.Logger(file_name=f'{config.log_dir}/terminallog.txt', file_mode="w+", should_flush=True)
    ngpus = torch.cuda.device_count()

    # load model
    print('starting the model and loader...')
    model = DensePure_Certify(args, config)
    if ngpus > 1:
        model = torch.nn.DataParallel(model)
    model = model.eval().to(device)

    # load dataset
    dataset = get_dataset(config.dataset.domain, 'test')

    # eval classifier and sde_adv against attacks
    purified_certify(model, dataset, args, config)


    logger.close()

def set_seed(seed: int=42)-> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    # diffusion models
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('--toolkit', action='store_true', help='whether to use run on toolkit')
    

    args = parser.parse_args()

    # parse config file
    yaml_path = os.path.join('configs', 'imagenet.yml')
    if args.toolkit : yaml_path = os.path.join('/mnt/home/DensePure', yaml_path)
    config = OmegaConf.load(yaml_path)

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    # set random seed
    set_seed(args.seed)

    return args, config


if __name__ == '__main__':
    args, config = parse_args_and_config()
    # print(args)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    robustness_eval(args, config, device)
