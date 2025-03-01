import os
import random

import torch
import torchvision.utils as tvu

from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
import math
import numpy as np

from config_path import PathConfig


class GuidedDiffusionFilterPerPixel(torch.nn.Module):
    def __init__(self, args, config, device=None, model_dir='pretrained'):
        super().__init__()
        self.args = args
        self.config = config
        self.budget_jump_to_guiding_ratio = float(config.certify.budget_jump_to_guiding_ratio) if config.certify.budget_jump_to_guiding_ratio!= 'inf' else np.inf
        self.guide_scale = float(config.scale)
        print(f"scale is {self.guide_scale}")
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device
        self.reverse_state = None
        self.reverse_state_cuda = None

        # self.reverse_seed = 1

        # load model
        model_config = model_and_diffusion_defaults()
        model_config.update(config.model)
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

        sigma = self.config.certify.sigma

        sigma = sigma*2

        # sigma_guide = ratio * sigma_jump
        jump_sigma = sigma * np.sqrt(1 + 1 / self.budget_jump_to_guiding_ratio ** 2) if self.budget_jump_to_guiding_ratio != 0 else sigma
        self.guide_sigma = sigma * np.sqrt(1 + self.budget_jump_to_guiding_ratio ** 2)
        print(f"jump sigma is {jump_sigma}, guide sigma is {self.guide_sigma}")
        print(np.sqrt(1/ (1 / jump_sigma ** 2 + 1/ self.guide_sigma ** 2)))

        a = 1/(1+(jump_sigma)**2)
        self.scale = a**0.5

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
            model_kwargs={"img" :  rescaled_original_img, "stop": 0}
            # print(f"original img is {rescaled_original_img.min()}, {rescaled_original_img.max()}")


            self.model.eval()

            # added Feb12
            # global_seed_state = torch.random.get_rng_state()
            # if torch.cuda.is_available():
            #     global_cuda_state = torch.cuda.random.get_rng_state_all()
            # if self.reverse_state==None:
            #     torch.manual_seed(self.reverse_seed)
            #     if torch.cuda.is_available():
            #         torch.cuda.manual_seed_all(self.reverse_seed)
            # else:
            #     torch.random.set_rng_state(self.reverse_state)
            #     if torch.cuda.is_available():
            #         torch.cuda.random.set_rng_state_all(self.reverse_state_cuda)


            # t steps denoise
            inter = t/self.config.num_t_steps # 396/10=39.6
            indices_t_steps = [round(t-i*inter) for i in range(self.config.num_t_steps)] #[396, 356, 317, 277, 238, 198, 158, 119, 79, 40]
            
            
            self.sq_guiding_budget = 1 / (self.guide_sigma * self.guide_sigma)
            # breakpoint()
            # print(f"total sq_budget is {self.sq_guiding_budget}")
            self.sq_budget = (torch.ones_like(img) * self.sq_guiding_budget)
            self.filter = torch.ones_like(img, dtype=torch.bool)


            for i in range(len(indices_t_steps)):
                t = torch.tensor([len(indices_t_steps)-i-1] * noisy_img.shape[0], device=self.device)
                real_t = torch.tensor([indices_t_steps[i]] * noisy_img.shape[0], device=self.device)
                print(f"at i={i}, t is {t[0].item()}, real_t is {real_t[0].item()}, step is {len(indices_t_steps)-i}")
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
                        step = len(indices_t_steps)-i,
                        real_t = real_t
                    )
                    noisy_img = out["sample"]
                    # print(f"x is {noisy_img.min():.3f}, {noisy_img.max():.3f}")

                # print(f"sq_budget is {self.sq_budget}")

                if torch.sum(self.filter) == 0:
                    model_kwargs["stop"] = real_t[0]
                    # print(f"stop at {real_t[0].item()}")
            

            # self.reverse_state = torch.random.get_rng_state()
            # if torch.cuda.is_available():
            #     # print("here as well")
            #     self.reverse_state_cuda = torch.cuda.random.get_rng_state_all()

            # torch.random.set_rng_state(global_seed_state)
            # if torch.cuda.is_available():
            #     # print("there as well")
            #     torch.cuda.random.set_rng_state_all(global_cuda_state)

            return noisy_img

    def cond_fn(self, x, t, **kwargs):
        scale = (torch.ones_like(x) * self.guide_scale)
        # breakpoint()
        scale[~self.filter] = 0
        # print(f"scale is {scale}")
        var = kwargs["var"]
        sqrt_alpha = kwargs["sqrt_alpha"]
        sqrt_alpha_t_minus_one = kwargs["sqrt_alpha_t_minus_one"]
        mean_t_minus_one = kwargs["mu_t"]
        rescaled_original_img = kwargs["img"]

        # print(f"alpha t coeff is {sqrt_alpha.min()}, {sqrt_alpha.max()}")
        # print(f"alpha t-1  coeff is {sqrt_alpha_t_minus_one.min()}, {sqrt_alpha_t_minus_one.max()}")
        # print(f"x is {x.min():.3f}, {x.max():.3f}")
        # print(f"x shape is {x.shape}")
        # print(f"img is {rescaled_original_img.min()}, {rescaled_original_img.max()}")
        # print(f"img shape is {rescaled_original_img.shape}")

        # accounting
        if self.config.guide_type == 'easy':
            mu_squared = scale * scale
        elif self.config.guide_type == 'alpha':
            mu_squared = scale * scale * sqrt_alpha * sqrt_alpha
        elif self.config.guide_type == 'mu':
            mu_squared = scale * scale * sqrt_alpha_t_minus_one * sqrt_alpha_t_minus_one
        else:
            raise Exception("error in guide_type, check config")

        if self.config.scaling_type == 'var_s':
            mu_squared = mu_squared * var
        elif self.config.scaling_type == 's':
            mu_squared = mu_squared / var
        else:
            raise Exception("error in scaling_type, check config")


        # print(f"current used mu square is : {mu_squared}")
        # print(f"sq_budget is {self.sq_budget}")
        out_of_budget_mask = mu_squared > self.sq_budget
        sqrt_alpha_tensor = (torch.ones_like(x) * sqrt_alpha)
        sqrt_alpha_t_minus_one_tensor = (torch.ones_like(x) * sqrt_alpha_t_minus_one)
    

        if self.config.scaling_type == 'var_s':
            scale[out_of_budget_mask] = torch.sqrt(self.sq_budget[out_of_budget_mask] / var[out_of_budget_mask])
        elif self.config.scaling_type == 's':
            scale[out_of_budget_mask] = torch.sqrt(self.sq_budget[out_of_budget_mask] * var[out_of_budget_mask])
        # else:
        #     raise Exception("error in scaling_type, check config")


        if self.config.guide_type == 'easy':
            guide = rescaled_original_img - x
        elif self.config.guide_type == 'alpha':
            guide = sqrt_alpha * rescaled_original_img - x
            scale[out_of_budget_mask] = scale[out_of_budget_mask] / sqrt_alpha_tensor[out_of_budget_mask]
        elif self.config.guide_type == 'mu':
            guide = sqrt_alpha_t_minus_one * rescaled_original_img - mean_t_minus_one
            scale[out_of_budget_mask] = scale[out_of_budget_mask] / sqrt_alpha_t_minus_one_tensor[out_of_budget_mask]
        # else:
        #     raise Exception("error in guide_type, check config")

        # print(f"guide value is {guide.min().item():.3f}, {guide.max().item():.3f}")


        self.sq_budget[out_of_budget_mask] = 0
        self.filter[out_of_budget_mask] = False
        self.sq_budget[~out_of_budget_mask] -= mu_squared[~out_of_budget_mask]
        if t[0] < kwargs["stop"] : scale = torch.zeros_like(x)


        if self.config.scaling_type == 'var_s':
            guide = guide  * scale if t[0]!= 0 else torch.zeros_like(x)

        elif self.config.scaling_type == 's':
            guide = guide  * scale / var if t[0]!= 0 else torch.zeros_like(x)

        print(f"guiding scale for step {t[0]}: {scale.max().item()}")

        # print(t[0].item())
        # breakpoint()
        # print(f"variance is {var.min().item():.3f}, {var.max().item():.3f}")
        print(f"guide value is {guide.min().item():.3f}, {guide.max().item():.3f}\n")
        return guide


