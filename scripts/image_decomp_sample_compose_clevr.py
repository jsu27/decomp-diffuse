import os
import numpy as np
import argparse
import torch as th

from composable_diffusion.download import load_checkpoint
from composable_diffusion.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler,
    add_dict_to_argparser,
    args_to_dict
)

from torchvision.utils import make_grid, save_image
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from imageio import imread
from skimage.transform import resize as imresize

has_cuda = th.cuda.is_available()
device = th.device('cpu' if not has_cuda else 'cuda')

options = model_and_diffusion_defaults()
options['use_fp16'] = has_cuda
options['timestep_respacing'] = '100'  # use 100 diffusion steps for fast sampling
options['num_classes'] = '2'

parser = argparse.ArgumentParser()
add_dict_to_argparser(parser, options)
parser.add_argument('--ckpt_path', required=True)
parser.add_argument('--latent_index', type=int)
parser.add_argument('--im_path', type=str, required=True)
parser.add_argument('--save_dir', type=str, default='clevr_gen_imgs')
parser.add_argument('--noise_std', type=float, default=1.0)


args = parser.parse_args()
ckpt_path = args.ckpt_path
del args.ckpt_path
# latent_index = args.latent_index
# del args.latent_index
im_path = args.im_path
del args.im_path
save_dir = args.save_dir
del args.save_dir
noise_std = args.noise_std
del args.noise_std
# save_dir = f'clevr_{options["image_size"]}_gen_imgs'


options = args_to_dict(args, model_and_diffusion_defaults().keys())
options['dataset'] = 'clevr' # decomp U-Net model
model, diffusion = create_model_and_diffusion(**options)

model.eval()
if has_cuda:
    model.convert_to_fp16()
model.to(device)

print(f'loading from {ckpt_path}')
checkpoint = th.load(ckpt_path, map_location='cpu')
model.load_state_dict(checkpoint)

print('total base parameters', sum(x.numel() for x in model.parameters()))


def show_images(batch: th.Tensor, file_name: str = 'result.png'):
    """Display a batch of images inline."""
    scaled = ((batch + 1) * 127.5).round().clamp(0, 255).to(th.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
    Image.fromarray(reshaped.numpy()).save(file_name)



"""
Get some real images.

"""
# eg get a Clevr image
im_path = 'im_10.png'
im = imread(im_path)
im = imresize(im, (64, 64))[:, :, :3]
im = th.Tensor(im).permute(2, 0, 1)[None, :, :, :].contiguous().cuda()
# im = imresize(im, (128, 128))
# im = torch.Tensor(im).permute(2, 0, 1)[None, :, :, :].contiguous().cuda()


batch_size = 1
guidance_scale = 10.0

# Tune this parameter to control the sharpness of 256x256 images.
# A value of 1.0 is sharper, but sometimes results in grainy artifacts.
# upsample_temp = 0.997
upsample_temp = 0.980

##############################
# Sample from the base model #
##############################

# # Create the position label
# positions = [[0.1, 0.5], [0.3, 0.5], [0.5, 0.5], [0.7, 0.5], [0.9, 0.5], [-1, -1]]  # horizontal
# full_batch_size = batch_size * len(positions)
# masks = [True] * (len(positions) - 1) + [False]

latent = model.embed_latent(im)
# model_kwargs['latent'] = latent # pass in latents of real imgs

model_kwargs = dict(
    latent=latent
    # y=th.tensor(positions, dtype=th.float, device=device),
    # masks=th.tensor(masks, dtype=th.bool, device=device),
    # latent_index=latent_index
)


# x: latent z1, z2, z3
# generate img with these latents passed in (concatenated)
# should recreate x

# Sample from the base model.
number_images = 4
all_samples = []

# save_dir = f'clevr_{options["image_size"]}_gen_imgs'

def gen_image(model, batch_size, options, device, model_kwargs, number_images=4, desc='', save_dir=''):
    all_samples = []

    # should get desired real imgs here
    for i in range(number_images):
        # latent = model.embed_latent()
        samples = diffusion.p_sample_loop(
            model,
            (batch_size, 3, options["image_size"], options["image_size"]),
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
            learn_sigma=options["learn_sigma"],
            noise_std=noise_std
        )[:batch_size]

        all_samples.append(samples)

    samples = th.cat(all_samples, dim=0).cpu() # did not need to rescale bc orig training imgs not rescaled
    # import pdb; pdb.set_trace()
    grid = make_grid(samples, nrow=int(samples.shape[0] ** 0.5), padding=0)
    if len(desc) > 0:
        desc = '_' + desc

    if len(save_dir) > 0:
        os.makedirs(save_dir, exist_ok=True)
        save_dir = save_dir + '/'
    save_image(grid, f'{save_dir}clevr_{options["image_size"]}_{guidance_scale}{desc}.png')

# should get desired real imgs here
gen_image(model, batch_size, options, device, model_kwargs, number_images=number_images, save_dir=save_dir)

# also want to sample w 1 latent at a time
num_comps = 3
latent_dim = latent.shape[1] // num_comps # length of single latent
for i in range(num_comps):
    model_kwargs['latent_index'] = i
    gen_image(model, batch_size, options, device, model_kwargs, number_images=number_images, desc=str(i), save_dir=save_dir)

