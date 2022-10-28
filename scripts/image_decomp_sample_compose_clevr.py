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

args = parser.parse_args()
ckpt_path = args.ckpt_path
del args.ckpt_path
# latent_index = args.latent_index
# del args.latent_index
im_path = args.im_path
del args.im_path

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
    )[:batch_size]

    all_samples.append(samples)


samples = ((th.cat(all_samples, dim=0) + 1) * 127.5).round().clamp(0, 255).to(th.uint8).cpu() / 255.
grid = make_grid(samples, nrow=int(samples.shape[0] ** 0.5), padding=0)
save_image(grid, f'clevr_{options["image_size"]}_{guidance_scale}.png')


def gen_image(model, batch_size, options, device, model_kwargs, number_images=4, desc=''):
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
        )[:batch_size]

        all_samples.append(samples)


    samples = ((th.cat(all_samples, dim=0) + 1) * 127.5).round().clamp(0, 255).to(th.uint8).cpu() / 255.
    grid = make_grid(samples, nrow=int(samples.shape[0] ** 0.5), padding=0)
    if len(desc) > 0:
        desc = '_' + desc
    save_image(grid, f'clevr_{options["image_size"]}_{guidance_scale}{desc}.png')


"""
also want to sample w 1 latent at a time
"""
num_comps = 3
latent_dim = latent.shape[1] // num_comps # length of single latent
for i in range(num_comps):
    # latent_comp = th.zeros(latent.shape).to(device)
    # latent_comp[:, i * latent_dim: (i+1) * latent_dim] = latent[:, i * latent_dim: (i+1) * latent_dim] # extract out latent from flattened latent vec
    # # latent_comp.to(device)
    # model_kwargs['latent'] = latent_comp
    model_kwargs['latent_index'] = i
    gen_image(model, batch_size, options, device, model_kwargs, number_images=number_images, desc=str(i))



# # import argparse
# # from models import LatentEBM128
# # from imageio import imread, get_writer
# # from skimage.transform import resize as imresize
# # import torch


# def gen_image(latents, FLAGS, models, im_neg, num_steps, idx=None):
#     """
#     latents: latent embeddings for concepts
#     FLAGS: .components: number of concepts
#     models: list of models per concept
#     """
#     im_negs = []

#     im_neg.requires_grad_(requires_grad=True)

#     for i in range(num_steps):
#         energy = 0

#         for j in range(len(latents)):
#             if idx is not None and idx != j:
#                 pass
#             else:
#                 ix = j % FLAGS.components
#                 energy = models[j % FLAGS.components].forward(im_neg, latents[j]) + energy

#         im_grad, = torch.autograd.grad([energy.sum()], [im_neg])

#         im_neg = im_neg - FLAGS.step_lr * im_grad

#         im_neg = torch.clamp(im_neg, 0, 1)
#         im_negs.append(im_neg)
#         im_neg = im_neg.detach()
#         im_neg.requires_grad_()

#     return im_negs


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Train EBM model')
#     parser.add_argument('--im_path', default='im_0.jpg', type=str, help='image to load')
#     args = parser.parse_args()

#     ckpt = torch.load("celebahq_128.pth")
#     FLAGS = ckpt['FLAGS']
#     state_dict = ckpt['model_state_dict_0']

#     model = LatentEBM128(FLAGS, 'celebahq_128').cuda()
#     model.load_state_dict(state_dict)
#     models = [model for i in range(4)]


#     im = imread(args.im_path)
#     im = imresize(im, (128, 128))
#     im = torch.Tensor(im).permute(2, 0, 1)[None, :, :, :].contiguous().cuda()

#     latent = model.embed_latent(im)
#     latents = torch.chunk(latent, 4, dim=1)

#     im_neg = torch.rand_like(im)

#     FLAGS.step_lr = 200.0
#     ims = gen_image(latents, FLAGS, models, im_neg, 30)

#     writer = get_writer("im_opt_full.mp4")
#     for im in ims:
#         im = im.detach().cpu().numpy()[0]
#         im = im.transpose((1, 2, 0))
#         writer.append_data(im)

#     writer.close()

#     for i in range(4):
#         writer = get_writer("im_opt_{}.mp4".format(i))

#         ims = gen_image(latents, FLAGS, models, im_neg, 30, idx=i)

#         for im in ims:
#             im = im.detach().cpu().numpy()[0]
#             im = im.transpose((1, 2, 0))
#             writer.append_data(im)

#         writer.close()
