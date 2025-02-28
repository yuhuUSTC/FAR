from functools import partial

import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import math
import torch
import torch.nn as nn

from timm.models.vision_transformer import Block
from diffusers.models.attention import BasicTransformerBlock
from models.diffloss import DiffLoss

import torch.nn.functional as F
import torchvision.transforms as T
import random
import os

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


def mask_by_order(mask_len, order, bsz, seq_len):
    masking = torch.zeros(bsz, seq_len).cuda()
    masking = torch.scatter(masking, dim=-1, index=order[:, :mask_len.long()], src=torch.ones(bsz, seq_len).cuda())#.bool()
    return masking


class FAR_T2I(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=256, vae_stride=16, patch_size=1,
                 encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
                 decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 vae_embed_dim=16,
                 mask_ratio_min=0.7,
                 label_drop_prob=0.1,
                 class_num=1000,
                 attn_dropout=0.1,
                 proj_dropout=0.1,
                 buffer_size=300,
                 diffloss_d=3,
                 diffloss_w=1024,
                 num_sampling_steps='100',
                 diffusion_batch_mul=4
                 ):
        super().__init__()

        # --------------------------------------------------------------------------
        # VAE and patchify specifics
        self.vae_embed_dim = vae_embed_dim

        self.img_size = img_size
        self.vae_stride = vae_stride
        self.patch_size = patch_size
        self.seq_h = self.seq_w = img_size // vae_stride // patch_size
        self.seq_len = self.seq_h * self.seq_w
        self.token_embed_dim = vae_embed_dim * patch_size**2
        self.loss_weight = [1 + np.sin(math.pi / 2. * (bands + 1) / self.seq_h) for bands in range(self.seq_h)]

        # --------------------------------------------------------------------------
        # Class Embedding
        self.num_classes = class_num
        #self.class_emb = nn.Embedding(1000, encoder_embed_dim)
        self.context_embed = nn.Linear(1536, encoder_embed_dim, bias=True)
        self.label_drop_prob = label_drop_prob
        # Fake class embedding for CFG's unconditional generation
        self.fake_latent = nn.Parameter(torch.zeros(1, 300, encoder_embed_dim))

        # --------------------------------------------------------------------------
        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25)

        # --------------------------------------------------------------------------
        self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)
        self.z_proj_ln = nn.LayerNorm(encoder_embed_dim, eps=1e-6)
        self.buffer_size = buffer_size
        self.encoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, encoder_embed_dim))

        self.encoder_blocks = nn.ModuleList([
            BasicTransformerBlock(
                encoder_embed_dim,
                encoder_num_heads,
                64,
                dropout=0.0,
                cross_attention_dim=encoder_embed_dim,
                activation_fn="gelu-approximate",
                attention_bias=True,
                upcast_attention=False,
                norm_type="layer_norm",
                norm_elementwise_affine=False,
                norm_eps=1e-5,
            ) for _ in range(encoder_depth)])
        self.encoder_norm = norm_layer(encoder_embed_dim)

        # --------------------------------------------------------------------------
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            BasicTransformerBlock(
                decoder_embed_dim,
                decoder_num_heads,
                64,
                dropout=0.0,
                cross_attention_dim=encoder_embed_dim,
                activation_fn="gelu-approximate",
                attention_bias=True,
                upcast_attention=False,
                norm_type="layer_norm",
                norm_elementwise_affine=False,
                norm_eps=1e-5,
            ) for _ in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len, decoder_embed_dim))

        self.initialize_weights()

        # --------------------------------------------------------------------------
        # Diffusion Loss
        self.diffloss = DiffLoss(
            target_channels=self.token_embed_dim,
            z_channels=decoder_embed_dim,
            width=diffloss_w,
            depth=diffloss_d,
            num_sampling_steps=num_sampling_steps,
        )
        self.diffusion_batch_mul = diffusion_batch_mul

    def initialize_weights(self):
        # parameters
        #torch.nn.init.normal_(self.class_emb.weight, std=.02)
        torch.nn.init.normal_(self.fake_latent, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.encoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.diffusion_pos_embed_learned, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        bsz, c, h, w = x.shape
        p = self.patch_size
        h_, w_ = h // p, w // p

        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(bsz, h_ * w_, c * p ** 2)
        return x  # [n, l, d]

    def unpatchify(self, x):
        bsz = x.shape[0]
        p = self.patch_size
        c = self.vae_embed_dim
        h_, w_ = self.seq_h, self.seq_w

        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x  # [n, c, h, w]

    def sample_orders(self, bsz):
        # generate a batch of random generation orders
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).cuda().long()
        return orders

    def random_masking(self, x, orders):
        # generate token mask
        bsz, seq_len, embed_dim = x.shape
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, seq_len, device=x.device)
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                             src=torch.ones(bsz, seq_len, device=x.device))
        return mask


    def processingpregt_latent(self, imgs):
        B, C, H, W = imgs.shape
        out = torch.zeros_like(imgs)
        latent_core = list(range(H))
        core_index = []

        for i in range(B):
            random_number = torch.randint(0, len(latent_core), (1,))
            chosen_core = latent_core[random_number]
            core_index.append(chosen_core)
            if random_number == 0:
                out[i] = torch.zeros(C, H, W).to(imgs.dtype).to(imgs.device)
            else:
                imgs_resize = F.interpolate(imgs[i].unsqueeze(0), size=(chosen_core, chosen_core), mode='area')
                out[i] = F.interpolate(imgs_resize, size=(H, W), mode='bicubic').squeeze(0)
        core_index = torch.tensor(core_index, device=out.device).to(imgs.dtype)
        return out, core_index


    def forward_mae_encoder_cross(self, x, mask, class_embedding):
        x = self.z_proj(x)
        bsz, seq_len, embed_dim = x.shape


        # random drop class embedding during training
        if self.training:
            drop_latent_mask = torch.rand(bsz) < self.label_drop_prob
            drop_latent_mask = drop_latent_mask.unsqueeze(1).unsqueeze(-1).cuda().to(x.dtype)
            class_embedding = drop_latent_mask * self.fake_latent + (1 - drop_latent_mask) * class_embedding

        # encoder position embedding
        x = x + self.encoder_pos_embed_learned
        x = self.z_proj_ln(x)

        # dropping
        x = x[(1-mask).nonzero(as_tuple=True)].reshape(bsz, -1, embed_dim)

        # apply Transformer blocks
        for blk in self.encoder_blocks:
            x = blk(
                x,
                attention_mask=None,
                encoder_hidden_states=class_embedding,
                encoder_attention_mask=None,
                timestep=None,
                cross_attention_kwargs=None,
                class_labels=None,
            )
        x = self.encoder_norm(x)

        return x

    def forward_mae_decoder_cross(self, x, mask, class_embedding):

        x = self.decoder_embed(x)

        # pad mask tokens
        mask_tokens = self.mask_token.repeat(mask.shape[0], mask.shape[1], 1).to(x.dtype)
        x_after_pad = mask_tokens.clone()
        x_after_pad[(1 - mask).nonzero(as_tuple=True)] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])

        # decoder position embedding
        x = x_after_pad + self.decoder_pos_embed_learned

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(
                x,
                attention_mask=None,
                encoder_hidden_states=class_embedding,
                encoder_attention_mask=None,
                timestep=None,
                cross_attention_kwargs=None,
                class_labels=None,
            )
        x = self.decoder_norm(x)

        x = x + self.diffusion_pos_embed_learned
        return x
    
    
    def forward_loss(self, z, target, mask, index, loss_weight=False):
        bsz, seq_len, _ = target.shape
        target = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        index = index.unsqueeze(1).unsqueeze(-1).repeat(1, seq_len, 1).reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bsz*seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        
        mask = mask.reshape(bsz*seq_len).repeat(self.diffusion_batch_mul)
        if loss_weight:
            loss_weight = loss_weight.unsqueeze(1).repeat(1, seq_len).reshape(bsz * seq_len).repeat(self.diffusion_batch_mul)
        loss = self.diffloss(z=z, target=target, index=index, loss_weight=loss_weight)
        return loss


    def forward(self, imgs, labels, loss_weight=False):
        class_embedding = self.context_embed(labels)

        process_imgs, x_index = self.processingpregt_latent(imgs)
        if loss_weight:
            loss_weight = self.loss_weight
        
        x = self.patchify(process_imgs)         # x.shape: torch.Size([128, 256, 16]))
        gt_latents = self.patchify(imgs)
        orders = self.sample_orders(bsz=x.size(0))
        mask = self.random_masking(x, orders)

        x = self.forward_mae_encoder_cross(x, mask, class_embedding)    
        z = self.forward_mae_decoder_cross(x, mask, class_embedding)     # z.shape: torch.Size([bs, 256, 1024])

        loss = self.forward_loss(z=z, target=gt_latents, mask=mask, index=x_index, loss_weight=loss_weight)
        
        return loss
    


    def sample_tokens(self, vae, bsz, num_iter=64, cfg=1.0, cfg_schedule="linear", labels=None, temperature=1.0, progress=False, device=None, output_dir=None):
        latent_core = [0,2,3,4,5,6,7,8,10,13]

        num_iter = len(latent_core)
        mask = torch.ones(bsz, self.seq_len).to(device)
        tokens = torch.zeros(bsz, self.seq_len, self.token_embed_dim).to(device)
        orders = self.sample_orders(bsz)

        class_embedding = self.context_embed(labels)
        class_embedding = torch.cat([class_embedding, self.fake_latent.repeat(bsz, 1, 1)], dim=0)

        for step in list(range(num_iter)):
            cur_tokens = tokens.clone()
            tokens = torch.cat([tokens, tokens], dim=0)
            
            mask = torch.cat([mask, mask], dim=0)
            
            x = self.forward_mae_encoder_cross(tokens, mask, class_embedding)
            z = self.forward_mae_decoder_cross(x, mask, class_embedding)    # torch.Size([512, 256, 768])    var输出的condition的维度很高(768)，var一次生成所有token后，只随机取部分(nge)送到diffusion中作为条件，生成部分token。
            B, L, C = z.shape
            z = z.reshape(B * L, -1)

            # mask ratio for the next round, following MaskGIT and MAGE.
            mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
            mask_len = torch.Tensor([np.floor(self.seq_len * mask_ratio)]).to(device)

            cfg_iter = 1.0 + (cfg - 1.0) * step / num_iter
            cfg_iter = cfg
            temperature_iter = 0.8 + (1 - np.cos(math.pi / 2. * (step + 1) / num_iter)) * (1-0.8)

            index = torch.tensor([latent_core[step]]).unsqueeze(1).unsqueeze(-1).repeat(B, L, 1).reshape(B * L, -1).to(device)
            sampled_token_latent = self.diffloss.sample(z, temperature_iter, cfg_iter, index, device)     # torch.Size([512, 16])
 
            z, _ = sampled_token_latent.chunk(2, dim=0)  # Remove null class samples.  torch.Size([256, 16])
            
            if step < num_iter-1:
                z = z.reshape(bsz, L, -1).transpose_(1, 2).reshape(bsz, -1, self.seq_h, self.seq_w)
                imgs_resize = F.interpolate(z, size=(latent_core[step+1], latent_core[step+1]), mode='area')
                z = F.interpolate(imgs_resize, size=(self.seq_h, self.seq_w), mode='bicubic')
                z = z.reshape(bsz, -1, L).transpose_(1, 2).reshape(bsz*L, -1)
            
            sampled_token = z.reshape(bsz, L, -1)
            mask_next = mask_by_order(mask_len[0], orders, bsz, self.seq_len)
            mask_to_pred = torch.logical_not(mask_next.bool())
            mask = mask_next
            sampled_token_latent = sampled_token[mask_to_pred.nonzero(as_tuple=True)]
            
            cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
            tokens = cur_tokens.clone()

        tokens = tokens.transpose_(1, 2).reshape(bsz, -1, self.seq_h, self.seq_w)
        tokens = vae.decode(tokens)

        return tokens
        

         
def far_t2i(**kwargs):
    model = FAR_T2I(
        encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
        decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
