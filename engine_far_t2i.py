import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched
from models.vae import DiagonalGaussianDistribution
import torch_fidelity
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import time
from models.tools import encode_prompts
import IPython
from torch.profiler import profile, record_function, ProfilerActivity

from torchvision.utils import make_grid
from typing import Optional
from PIL import Image
def save_image(images: torch.Tensor, nrow: int = 8, show: bool = True, path: Optional[str] = None, format: Optional[str] = None, to_grayscale: bool = False, **kwargs):
    images = images * 0.5 + 0.5
    grid = make_grid(images, nrow=nrow, **kwargs)  # (channels, height, width)
    #  (height, width, channels)
    grid = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(grid)
    if to_grayscale:
        im = im.convert(mode="L")
    if path is not None:
        im.save(path, format=format)
    if show:
        im.show()
    return grid


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def train_one_epoch(text_tokenizer, text_model, llm_system_prompt, model, vae,
                    model_params, ema_params,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))


    for data_iter_step, (samples, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        
        with torch.no_grad():
            labels = encode_prompts(
                labels,
                text_model,
                text_tokenizer,
                text_tokenizer_max_length=300,
                system_prompt=llm_system_prompt,
                use_llm_system_prompt=True,
            )
        
        with torch.no_grad():
            x = vae.encode(samples).sample().mul_(0.2325)

        
        with torch.cuda.amp.autocast():
            loss = model(x, labels, loss_weight=args.loss_weight)
        loss_value = loss.item()


        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss_scaler(loss, optimizer, clip_grad=args.grad_clip, parameters=model.parameters(), update_grad=True)
        optimizer.zero_grad()

        torch.cuda.synchronize()

        update_ema(ema_params, model_params, rate=args.ema_rate)

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and int(os.environ.get('RANK'))==0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(text_tokenizer, text_model, llm_system_prompt, model_without_ddp, vae, ema_params, args, epoch, batch_size=16, log_writer=None, cfg=1.0,
             use_ema=True):
    model_without_ddp.eval()
    num_steps = 1

    # switch to ema params
    if use_ema:
        model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
            assert name in ema_state_dict
            ema_state_dict[name] = ema_params[i]
        print("Switch to ema")
        model_without_ddp.load_state_dict(ema_state_dict)

    used_time = 0
    gen_img_cnt = 0

    
    for i in range(num_steps):
        print("Generation step {}/{}".format(i, num_steps))
        
        prompts = ["A photo of a dog", "A happy girl", "A boy and a girl fall in love.", "Drone view of waves crashing against the rugged cliffs in Big Sur ..", 
                   "A dog that has been meditating all the time",  "Editorial photoshoot of a old woman, high fashion 2000s fashion", "A small cactus with a happy face in the Sahara desert.", "An astronaut riding a horse on the moon, oil painting by Van Gogh.", 
                  "a photo of a bear next to a STOP sign", "Mona Lisa wearing headphone", "Cute small dog sitting in a movie theater eating popcorn watching a movie", "an eggplant and an avocado, stuffed toys next to each other",
                  "Someone is holding a phone while watching a video on it.", "so many people using laptops at a meeting", "this is a very dark picture of a room with a shelf", "A wooden bed with a brown dog sleeping on top of it.", 
                  "a bed sits under neath a glass fish tank ", "A shower has a removable shower head and a glass door.", "The two figurines and bells are under the clock.", "A pond of water with three giraffe walking in the dirt.", 
                  "Fast commuter train moving past an outdoor platform.", "The police officers are sitting together and eating food.", "A laptop and a cell phone on a table.", "A black dog in a bathroom drinking out of the toilet.",
                  "A photo of a smiling person with snow goggles onholding a snowboard", "A photo of a cat playing chess.", "A bird made of crystal", "A pair of old boots covered in mud.", 
                   "Photo of a bear catching salmon.",  "High quality, a close up photo of a human hand", "A white horse reading a book, fairytale.", "A window with raindrops tricklingdown, overlooking a blurry city."]

        
        with torch.no_grad():
            labels_gen = encode_prompts(
                prompts,
                text_model,
                text_tokenizer,
                text_tokenizer_max_length=300,
                system_prompt=llm_system_prompt,
                use_llm_system_prompt=True,
            )


        torch.cuda.synchronize()
        start_time = time.time()

        device = torch.device("cuda")
        # generation
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                sampled_images = model_without_ddp.sample_tokens(vae, bsz=batch_size, num_iter=args.num_iter, cfg=cfg,
                                                                 cfg_schedule=args.cfg_schedule, labels=labels_gen, device=device,
                                                                 temperature=args.temperature, output_dir=args.output_dir)

        # measure speed after the first generation batch
        if i >= 1:
            torch.cuda.synchronize()
            used_time += time.time() - start_time
            gen_img_cnt += batch_size
            print("Generating {} images takes {:.5f} seconds, {:.5f} sec per image".format(gen_img_cnt, used_time, used_time / gen_img_cnt))

        torch.distributed.barrier()
        sampled_images = sampled_images.detach().cpu()

        save_image(sampled_images, nrow=8, show=False, path=os.path.join(args.output_dir, f"epoch{epoch}.png"), to_grayscale=False)

    torch.distributed.barrier()
    time.sleep(10)

    # back to no ema
    if use_ema:
        print("Switch back from ema")
        model_without_ddp.load_state_dict(model_state_dict)




        
def cache_latents(vae,
                  data_loader: Iterable,
                  device: torch.device,
                  args=None):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Caching: '
    print_freq = 20

    for data_iter_step, (samples, _, paths) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        samples = samples.to(device, non_blocking=True)

        with torch.no_grad():
            posterior = vae.encode(samples)
            moments = posterior.parameters
            posterior_flip = vae.encode(samples.flip(dims=[3]))
            moments_flip = posterior_flip.parameters

        for i, path in enumerate(paths):
            save_path = os.path.join(args.cached_path, path + '.npz')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez(save_path, moments=moments[i].cpu().numpy(), moments_flip=moments_flip[i].cpu().numpy())

        if misc.is_dist_avail_and_initialized():
            torch.cuda.synchronize()

    return
