import os
import random

import torch
import torchvision.utils as tvu

from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
import math
import numpy as np

from config_path import PathConfig


class GuidedDiffusion(torch.nn.Module):
    def __init__(self, args, config, device=None, model_dir='pretrained'):
        super().__init__()
        self.args = args
        self.config = config
        self.budget_jump_to_guiding_ratio = float(args.budget_jump_to_guiding_ratio) if args.budget_jump_to_guiding_ratio!= 'inf' else np.inf
        self.guide_scale = float(args.scale)
        print(f"scale is {self.guide_scale}")
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device
        self.reverse_state = None
        self.reverse_state_cuda = None

        # load model
        model_config = model_and_diffusion_defaults()
        model_config.update(vars(self.config.model))
        # print(f'model_config: {model_config}')
        model, diffusion = create_model_and_diffusion(**model_config)
        # model.load_state_dict(torch.load(f'{model_dir}/256x256_diffusion_uncond.pt', map_location='cpu'))
        model_path = PathConfig().get_model_path()
        model.load_state_dict(torch.load(model_path))
        model.requires_grad_(False).eval().to(self.device)

        if model_config['use_fp16']:
            model.convert_to_fp16()

        self.model = model
        self.model.eval()
        self.diffusion = diffusion
        self.betas = diffusion.betas
        alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        sigma = self.args.sigma

        sigma = sigma*2

        # sigma_guide = ratio * sigma_jump
        jump_sigma = sigma * np.sqrt(1 + 1 / self.budget_jump_to_guiding_ratio ** 2) if self.budget_jump_to_guiding_ratio != 0 else sigma
        self.guide_sigma = sigma * np.sqrt(1 + self.budget_jump_to_guiding_ratio ** 2)
        print(f"jump sigma is {jump_sigma}, guide sigma is {self.guide_sigma}")
        print(np.sqrt(1/ (1 / jump_sigma ** 2 + 1/ self.guide_sigma ** 2)))

        a = 1/(1+(jump_sigma)**2)
        self.scale = a**0.5

        T = self.args.t_total
        for t in range(len(self.sqrt_recipm1_alphas_cumprod)-1):
            if self.sqrt_recipm1_alphas_cumprod[t]<jump_sigma and self.sqrt_recipm1_alphas_cumprod[t+1]>=jump_sigma:
                if jump_sigma - self.sqrt_recipm1_alphas_cumprod[t] > self.sqrt_recipm1_alphas_cumprod[t+1] - jump_sigma:
                    self.t = t+1
                    break
                else:
                    self.t = t
                    break
            self.t = len(diffusion.alphas_cumprod)-1
        print(f"jump to step {t}")

    def image_editing_sample(self, img=None, original_x=None, bs_id=0, tag=None):
        assert isinstance(img, torch.Tensor)
        batch_size = img.shape[0]

        with torch.no_grad():
            # if tag is None:
            #     tag = 'rnd' + str(random.randint(0, 10000))
            # out_dir = os.path.join(self.args.log_dir, 'bs' + str(bs_id) + '_' + tag)

            assert img.ndim == 4, img.ndim

            noisy_img = self.scale*(img) #img is already added noise in core.py
            t = self.t
            # print(f"input img is {img.min():.3f}, {img.max():.3f}")

            rescaled_original_img = 2 * original_x - 1

            model_kwargs={"img" :  rescaled_original_img}
            # print(f"original img is {rescaled_original_img.min()}, {rescaled_original_img.max()}")

            if self.args.use_clustering:
                noisy_img = noisy_img.unsqueeze(1).repeat(1,self.args.clustering_batch,1,1,1).view(batch_size*self.args.clustering_batch,3,256,256)
            self.model.eval()

            if self.args.use_one_step:
                # one step denoise
                t = torch.tensor([round(t)] * noisy_img.shape[0], device=self.device)
                out = self.diffusion.p_sample(
                    self.model,
                    noisy_img,
                    t+self.args.t_plus,
                    clip_denoised=True,
                )
                noisy_img = out["pred_xstart"]

            elif self.args.use_t_steps:
                # t steps denoise
                inter = t/self.args.num_t_steps # 396/10=39.6
                indices_t_steps = [round(t-i*inter) for i in range(self.args.num_t_steps)] #[396, 356, 317, 277, 238, 198, 158, 119, 79, 40]
                
                for i in range(len(indices_t_steps)):
                    t = torch.tensor([len(indices_t_steps)-i-1] * noisy_img.shape[0], device=self.device)
                    real_t = torch.tensor([indices_t_steps[i]] * noisy_img.shape[0], device=self.device)
                    # print(f" at i={i}, t is {t[0].item()}, real_t is {real_t[0].item()}, step is {len(indices_t_steps)-i}")
                    # at i=0, t is 9, real_t is 396, step is 10
                    # at i=1, t is 8, real_t is 356, step is 9
                    # at i=2, t is 7, real_t is 317, step is 8
                    # at i=3, t is 6, real_t is 277, step is 7
                    # at i=4, t is 5, real_t is 238, step is 6
                    # at i=5, t is 4, real_t is 198, step is 5
                    # at i=6, t is 3, real_t is 158, step is 4
                    # at i=7, t is 2, real_t is 119, step is 3
                    # at i=8, t is 1, real_t is 79, step is 2
                    # at i=9, t is 0, real_t is 40, step is 1
                    # print(f"t before is {t[0].item()}")
                    with torch.no_grad():
                        out = self.diffusion.p_sample(
                            self.model,
                            noisy_img,
                            t,
                            clip_denoised=True,
                            cond_fn = self.cond_fn,
                            model_kwargs = model_kwargs,
                            indices_t_steps = indices_t_steps.copy(),
                            T = self.args.t_total,
                            step = len(indices_t_steps)-i,
                            real_t = real_t
                        )
                        noisy_img = out["sample"]
                        # print(f"x is {noisy_img.min():.3f}, {noisy_img.max():.3f}")
            else:
                # full steps denoise
                indices = list(range(round(t)))[::-1]
                for i in indices:
                    t = torch.tensor([i] * noisy_img.shape[0], device=self.device)
                    with torch.no_grad():
                        out = self.diffusion.p_sample(
                            self.model,
                            noisy_img,
                            t,
                            clip_denoised=True,
                        )
                        noisy_img = out["sample"]

            return noisy_img

    # def cond_fn(self, x, t, **kwargs):
    #     # scale = 2 * torch.ones(10).cuda()
    #     scale = 1 / (np.sqrt(self.args.num_t_steps) * self.guide_sigma)
    #     # print(f"scale is {scale:.3f}")
    #     var = kwargs["var"]
    #     alpha = kwargs["alpha"]
    #     rescaled_original_img = kwargs["img"]
    #     # print(f"x is {x.min():.3f}, {x.max():.3f}")
    #     # print(f"x shape is {x.shape}")
    #     # print(f"img is {rescaled_original_img.min()}, {rescaled_original_img.max()}")
    #     # print(f"img shape is {rescaled_original_img.shape}")
    #     guide = (alpha * rescaled_original_img - x) * scale / torch.sqrt(var) if t[0]!= 0 else torch.zeros_like(x)
    #     # print(t[0].item())
    #     # breakpoint()
    #     # print(f"variance is {var.min().item():.3f}, {var.max().item():.3f}")
    #     # print(f"guide value is {guide.min().item():.3f}, {guide.max().item():.3f}\n")
    #     return guide

    def cond_fn(self, x, t, **kwargs):
        # scale = 2 * torch.ones(10).cuda()
        scale = self.guide_scale
        # print(f"scale is {scale}")
        var = kwargs["var"]
        rescaled_original_img = kwargs["img"]
        # print(f"x is {x.min():.3f}, {x.max():.3f}")
        # print(f"x shape is {x.shape}")
        # print(f"img is {rescaled_original_img.min()}, {rescaled_original_img.max()}")
        # print(f"img shape is {rescaled_original_img.shape}")
        guide = (rescaled_original_img - x) * scale if t[0]!= 0 else torch.zeros_like(x)
        # print(t[0].item())
        # breakpoint()
        # print(f"variance is {var.min().item():.3f}, {var.max().item():.3f}")
        # print(f"guide value is {guide.min().item():.3f}, {guide.max().item():.3f}\n")
        return guide


