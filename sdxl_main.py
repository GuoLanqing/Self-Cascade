import os
import yaml
import random
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from transformers import CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModel, CLIPTextModelWithProjection
from diffusers import AutoencoderKL, DDPMScheduler
from source_override.unet2dconditionalmodel import UNet2DConditionModel
from ip_adapter.utils import is_torch2_available
from source_override.pipeline_sdxl_changed import StableDiffusionXLPipeline
if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor


def resize_image_keeping_aspect_ratio(image, target_shorter_side=2048):
    original_width, original_height = image.size

    # Calculate the aspect ratio preserving scale
    if original_width < original_height:
        new_width = target_shorter_side
        new_height = int((original_height / original_width) * target_shorter_side)
    else:
        new_height = target_shorter_side
        new_width = int((original_width / original_height) * target_shorter_side)

    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    return resized_image

class MyDataset(torch.utils.data.Dataset):

    def __init__(self, json_file, tokenizer, tokenizer_2, size=1024, center_crop=False, t_drop_rate=0.05,
                 i_drop_rate=0.05, ti_drop_rate=0.05, image_root_path=""):
        super().__init__()

        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.size = size
        self.center_crop = center_crop
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.image_root_path = image_root_path

        self.data = json.load(open(json_file))  # list of dict: [{"image_file": "1.png", "text": "A dog"}]

        self.transform = transforms.Compose([
            # transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        # self.clip_image_processor = CLIPImageProcessor()

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        image_file = item["image_file"]

        # read image
        raw_image = Image.open(os.path.join(self.image_root_path, image_file))

        resized_image = resize_image_keeping_aspect_ratio(raw_image, target_shorter_side=self.size)

        # original size
        original_width, original_height = raw_image.size
        original_size = torch.tensor([original_height, original_width])

        image_tensor = self.transform(resized_image.convert("RGB"))

        # random crop
        delta_h = image_tensor.shape[1] - self.size
        delta_w = image_tensor.shape[2] - self.size
        assert not all([delta_h, delta_w])

        if self.center_crop:
            top = delta_h // 2
            left = delta_w // 2
        else:
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        image = transforms.functional.crop(
            image_tensor, top=top, left=left, height=self.size, width=self.size
        )
        crop_coords_top_left = torch.tensor([top, left])

        # clip_image = self.clip_image_processor(images=raw_image, return_tensors="pt").pixel_values

        # drop
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            text = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            text = ""
            drop_image_embed = 1

        # get text and tokenize
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        text_input_ids_2 = self.tokenizer_2(
            text,
            max_length=self.tokenizer_2.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        return {
            "image": image,
            "prompt": text,
            "text_input_ids": text_input_ids,
            "text_input_ids_2": text_input_ids_2,
            # "clip_image": clip_image,
            "drop_image_embed": drop_image_embed,
            "original_size": original_size,
            "crop_coords_top_left": crop_coords_top_left,
            "target_size": torch.tensor([self.size, self.size]),
        }

    def __len__(self):
        return len(self.data)


def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    prompts = [example["prompt"] for example in data]  # Keep as list of string
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    text_input_ids_2 = torch.cat([example["text_input_ids_2"] for example in data], dim=0)
    # clip_images = torch.cat([example["clip_image"] for example in data], dim=0)
    drop_image_embeds = [example["drop_image_embed"] for example in data]
    original_size = torch.stack([example["original_size"] for example in data])
    crop_coords_top_left = torch.stack([example["crop_coords_top_left"] for example in data])
    target_size = torch.stack([example["target_size"] for example in data])

    return {
        "images": images,
        "prompts": prompts,
        "text_input_ids": text_input_ids,
        "text_input_ids_2": text_input_ids_2,
        # "clip_images": clip_images,
        "drop_image_embeds": drop_image_embeds,
        "original_size": original_size,
        "crop_coords_top_left": crop_coords_top_left,
        "target_size": target_size,
    }

# class IPAdapter(torch.nn.Module):
#     """IP-Adapter"""
#     def __init__(self, unet, image_proj_model, adapter_modules, ckpt_path=None):
#         super().__init__()
#         self.unet = unet
#         # self.image_proj_model = image_proj_model
#         # self.adapter_modules = adapter_modules
#         self.temp_downsample = nn.Upsample(scale_factor=0.5, mode='bilinear')
#
#         if ckpt_path is not None:
#             self.load_from_checkpoint(ckpt_path)
#
#     def forward(self, noisy_latents, timesteps, encoder_hidden_states):
#
#         # Predict the noise residual
#         lr_fea = self.temp_downsample(noisy_latents)
#         noise_pred = self.unet(sample=noisy_latents, lr_fea=lr_fea, timestep=timesteps, encoder_hidden_states=encoder_hidden_states).sample
#         return noise_pred
#
#     def load_from_checkpoint(self, ckpt_path: str):
#         # Calculate original checksums
#         orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
#         orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))
#
#         state_dict = torch.load(ckpt_path, map_location="cpu")
#
#         # Load state dict for image_proj_model and adapter_modules
#         self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
#         self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)
#
#
#         # Calculate new checksums
#         new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
#         new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))
#
#         # Verify if the weights have changed
#         assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
#         assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"
#
#         print(f"Successfully loaded weights from checkpoint {ckpt_path}")

class SelfCascadeModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        # Load models
        # self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(config['image_encoder_path'])
        self.noise_scheduler = DDPMScheduler.from_pretrained(config['pretrained_model_name_or_path'], subfolder="scheduler")
        self.text_encoder = CLIPTextModel.from_pretrained(config['pretrained_model_name_or_path'], subfolder="text_encoder")
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(config['pretrained_model_name_or_path'], subfolder="text_encoder_2")
        self.vae = AutoencoderKL.from_pretrained(config['pretrained_model_name_or_path'], subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(config['pretrained_model_name_or_path'], low_cpu_mem_usage=False, device_map=None, subfolder="unet")

        # self.pipeline = StableDiffusionXLPipeline.from_pretrained(
        #     "stabilityai/stable-diffusion-xl-base-1.0", output_type="latent", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        # ).to(self.device)

        # Freeze parameters
        # self.image_encoder.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.vae.requires_grad_(False)
        # self.unet.requires_grad_(False)

        trainable_modules = [
            'feature_upsampler0',
            'feature_upsampler1',
            'feature_upsampler2',
            'feature_upsampler3'
        ]
        for name, param in self.unet.named_parameters():
            if any(module in name for module in trainable_modules):
                param.requires_grad = True
            else:
                param.requires_grad = False

        a = 0


    def setup(self, stage=None):
        # Initialize the pipeline in setup method
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            output_type="latent",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        ).to(self.device)

    # def _init_attn_processors(self):
    #     attn_procs = {}
    #     unet_sd = self.unet.state_dict()
    #     for name in self.unet.attn_processors.keys():
    #         cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
    #         if name.startswith("mid_block"):
    #             hidden_size = self.unet.config.block_out_channels[-1]
    #         elif name.startswith("up_blocks"):
    #             block_id = int(name[len("up_blocks.")])
    #             hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
    #         elif name.startswith("down_blocks"):
    #             block_id = int(name[len("down_blocks.")])
    #             hidden_size = self.unet.config.block_out_channels[block_id]
    #         if cross_attention_dim is None:
    #             attn_procs[name] = AttnProcessor()
    #         else:
    #             layer_name = name.split(".processor")[0]
    #             weights = {
    #                 "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
    #                 "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
    #             }
    #             attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, num_tokens=4)
    #             attn_procs[name].load_state_dict(weights)
    #     return attn_procs

    def training_step(self, batch):
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.no_grad():
            _, lr_fea = self.pipeline(batch["prompts"])
            lr_fea = lr_fea.detach().float()


        # Encode images to latent space
        latents = self.vae.encode(batch["images"]).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        # Add noise to latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Encode images and text
        # image_embeds = self.image_encoder(batch["clip_images"]).image_embeds
        # image_embeds = [torch.zeros_like(embed) if drop else embed for embed, drop in zip(image_embeds, batch["drop_image_embeds"])]
        # image_embeds = torch.stack(image_embeds)

        encoder_output = self.text_encoder(batch['text_input_ids'], output_hidden_states=True)
        text_embeds = encoder_output.hidden_states[-2]
        encoder_output_2 = self.text_encoder_2(batch['text_input_ids_2'], output_hidden_states=True)
        pooled_text_embeds = encoder_output_2[0]
        text_embeds_2 = encoder_output_2.hidden_states[-2]
        text_embeds = torch.concat([text_embeds, text_embeds_2], dim=-1)

        # Prepare additional conditioning
        add_time_ids = torch.cat([
            batch["original_size"],
            batch["crop_coords_top_left"],
            batch["target_size"],
        ], dim=1)
        unet_added_cond_kwargs = {"text_embeds": pooled_text_embeds, "time_ids": add_time_ids}

        # Predict noise
        # lr_fea = self.temp_downsample(noisy_latents)
        lr_fea = lr_fea * self.vae.config.scaling_factor
        noise_pred = self.unet(sample=noisy_latents, lr_fea=lr_fea, timestep=timesteps,
                               encoder_hidden_states=text_embeds, added_cond_kwargs=unet_added_cond_kwargs).sample

        # Calculate loss
        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        print(f"Step {self.global_step}, Loss: {loss.item():.6f}")
        return loss

    def configure_optimizers(self):
        params_to_optimize = list(self.unet.feature_upsampler0.parameters()) + list(self.unet.feature_upsampler1.parameters()) + list(self.unet.feature_upsampler2.parameters()) + list(self.unet.feature_upsampler3.parameters())
        optimizer = torch.optim.AdamW(params_to_optimize, lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        return optimizer

class SelfCascadeDataModule(pl.LightningDataModule):
    def __init__(self, config ):
        super().__init__()
        self.config = config
        self.tokenizer = CLIPTokenizer.from_pretrained(config['pretrained_model_name_or_path'], subfolder="tokenizer")
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(config['pretrained_model_name_or_path'], subfolder="tokenizer_2")

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = MyDataset(
                self.config['data_json_file'],
                tokenizer=self.tokenizer,
                tokenizer_2=self.tokenizer_2,
                size=self.config['resolution'],
                image_root_path=self.config['data_root_path']
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config['train_batch_size'],
            shuffle=True,
            num_workers=self.config['dataloader_num_workers'],
            collate_fn=collate_fn
        )

def main():
    # Load configuration
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Initialize model and data module
    # if config['resume_checkpoint_path'] is not None:
    #     model = SelfCascadeModel.load_from_checkpoint(config['resume_checkpoint_path'], strict=True)
    # else:
    model = SelfCascadeModel(config)

    # model = SelfCascadeModel(config)
    data_module = SelfCascadeDataModule(config)

    # Set up logger and callbacks
    # save_model_name = args.name
    # save_path = f"./experiments_lightning/{save_model_name}/{args.version}"
    logger = TensorBoardLogger(save_dir=config['output_dir'], name='logs')
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['output_dir'],
        filename='checkpoint-{step}',
        save_top_k=-1,
        monitor='step',
        mode='max',
        every_n_train_steps=config['save_n_steps'],
        save_on_train_epoch_end=False,
        save_last=True,
        save_weights_only=False
    )

    # Initialize Trainer
    trainer = pl.Trainer(
        max_steps=config['max_steps'],
        # max_epochs=config['num_train_epochs'],
        logger=logger,
        log_every_n_steps=5,
        callbacks=[checkpoint_callback],
        precision='bf16',
        accelerator='gpu',
        devices=config['gpu_ids'],
        strategy=DDPStrategy(gradient_as_bucket_view=True, find_unused_parameters=True),
    )

    # Start training
    trainer.fit(model, data_module, ckpt_path=config['resume_checkpoint_path'])

if __name__ == "__main__":
    main()