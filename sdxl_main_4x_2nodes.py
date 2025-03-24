import os
import time
import json
import random
import torch
import torch.nn.functional as F
import numpy as np

torch.set_float32_matmul_precision('medium')
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.strategies import DDPStrategy, FSDPStrategy
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from transformers import (
    CLIPTokenizer,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from diffusers import AutoencoderKL, DDPMScheduler

from source_override.unet2dconditionalmodel import UNet2DConditionModel
from source_override.unet2dconditionalmodel_upsample import UNet2DConditionModel_upsample
from source_override.pipeline_sdxl_changed import StableDiffusionXLPipeline

from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
from diffusers.models.attention_processor import Attention
from diffusers.models.transformers.dual_transformer_2d import DualTransformer2DModel
from diffusers.models.transformers.transformer_2d import Transformer2DModel

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path


# ---------------------------
# EMA Callback Implementation
# ---------------------------
class EMACallback(pl.Callback):
    def __init__(self, model_key="unet", decay=0.9999):
        """
        Args:
            model_key (str): The attribute name of the model module to track (e.g., "unet").
            decay (float): The EMA decay rate.
        """
        super().__init__()
        self.model_key = model_key
        self.decay = decay
        self.ema_state_dict = {}

    def on_train_start(self, trainer, pl_module):
        # Copy initial parameters from the target module to initialize EMA weights.
        target_module = getattr(pl_module, self.model_key)
        self.ema_state_dict = {
            name: param.clone().detach() for name, param in target_module.named_parameters() if param.requires_grad
        }
        # Ensure EMA parameters are on the same device as the model.
        for name, ema_param in self.ema_state_dict.items():
            self.ema_state_dict[name] = ema_param.to(pl_module.device)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        target_module = getattr(pl_module, self.model_key)
        with torch.no_grad():
            for name, param in target_module.named_parameters():
                if param.requires_grad:
                    # Update the EMA parameter: ema = decay * ema + (1-decay) * param
                    self.ema_state_dict[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    # Optional: a helper method to apply EMA weights to the model for evaluation/inference.
    def apply_ema_weights(self, pl_module):
        target_module = getattr(pl_module, self.model_key)
        for name, param in target_module.named_parameters():
            if name in self.ema_state_dict:
                param.data.copy_(self.ema_state_dict[name])


# ---------------------------
# Time Checkpoint Callback
# ---------------------------
class TimeCheckpointCallback(pl.Callback):
    def __init__(self, checkpoint_path="temp.ckpt", run_duration=24 * 3600):
        """
        Args:
            checkpoint_path (str): Path to save the checkpoint.
            run_duration (int): Duration in seconds after which to save the checkpoint (default 24 hours).
        """
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.run_duration = run_duration  # 24 hours = 86400 seconds
        self.start_time = None
        self.checkpoint_saved = False

    def on_train_start(self, trainer, pl_module):
        # Record the start time of training.
        if self.start_time is None:
            self.start_time = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        # If the checkpoint hasn't been saved yet, check if elapsed time >= 24h.
        if not self.checkpoint_saved:
            elapsed = time.time() - self.start_time
            if elapsed >= self.run_duration:
                trainer.save_checkpoint(self.checkpoint_path)
                self.checkpoint_saved = True
                print(f"Checkpoint saved to {self.checkpoint_path} after {elapsed / 3600:.2f} hours of training.")


# ---------------------------
# Data Processing and Model Setup
# ---------------------------
def resize_image_keeping_aspect_ratio(image, target_shorter_side=2048):
    original_width, original_height = image.size
    if original_width < original_height:
        new_width = target_shorter_side
        new_height = int((original_height / original_width) * target_shorter_side)
    else:
        new_height = target_shorter_side
        new_width = int((original_width / original_height) * target_shorter_side)
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    return resized_image


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, json_file, tokenizer, tokenizer_2, size=1024, center_crop=False,
                 t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05, image_root_path=""):
        super().__init__()
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.size = size
        self.center_crop = center_crop
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.image_root_path = image_root_path

        # Load the JSON file containing a list of dicts
        self.data = json.load(open(json_file))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        image_file = item["image_file"]

        # Read and resize image
        raw_image = Image.open(os.path.join(self.image_root_path, image_file))
        resized_image = resize_image_keeping_aspect_ratio(raw_image, target_shorter_side=self.size)

        # Store original image size
        original_width, original_height = raw_image.size
        original_size = torch.tensor([original_height, original_width])

        image_tensor = self.transform(resized_image.convert("RGB"))

        # Random crop
        delta_h = image_tensor.shape[1] - self.size
        delta_w = image_tensor.shape[2] - self.size
        assert not all([delta_h, delta_w]), "Image is smaller than the crop size."

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

        # Randomly drop image and/or text conditions
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            text = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            text = ""
            drop_image_embed = 1

        # Tokenize text inputs
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
            "drop_image_embeds": drop_image_embed,
            "original_size": original_size,
            "crop_coords_top_left": crop_coords_top_left,
            "target_size": torch.tensor([self.size, self.size]),
        }

    def __len__(self):
        return len(self.data)


def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    prompts = [example["prompt"] for example in data]  # List of strings
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    text_input_ids_2 = torch.cat([example["text_input_ids_2"] for example in data], dim=0)
    drop_image_embeds = [example["drop_image_embeds"] for example in data]
    original_size = torch.stack([example["original_size"] for example in data])
    crop_coords_top_left = torch.stack([example["crop_coords_top_left"] for example in data])
    target_size = torch.stack([example["target_size"] for example in data])

    return {
        "images": images,
        "prompts": prompts,
        "text_input_ids": text_input_ids,
        "text_input_ids_2": text_input_ids_2,
        "drop_image_embeds": drop_image_embeds,
        "original_size": original_size,
        "crop_coords_top_left": crop_coords_top_left,
        "target_size": target_size,
    }


class SelfCascadeModel(pl.LightningModule):
    def __init__(self, opt: DictConfig):
        super().__init__()
        self.opt = opt
        self.save_hyperparameters()

        # Load components
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            opt.pretrained_model_name_or_path, subfolder="scheduler"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            opt.pretrained_model_name_or_path, subfolder="text_encoder"
        )
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            opt.pretrained_model_name_or_path, subfolder="text_encoder_2"
        )
        self.vae = AutoencoderKL.from_pretrained(
            opt.pretrained_model_name_or_path, subfolder="vae"
        )
        self.vae.enable_tiling()

        if opt.upsample_lr_fea == 'deconv':
            print("use deconv")
            self.unet = UNet2DConditionModel_upsample.from_pretrained(
                opt.pretrained_model_name_or_path, low_cpu_mem_usage=False, device_map=None, subfolder="unet"
            )
        else:
            print("use bilinear")
            self.unet = UNet2DConditionModel.from_pretrained(
                opt.pretrained_model_name_or_path, low_cpu_mem_usage=False, device_map=None, subfolder="unet"
            )
        self.unet.enable_xformers_memory_efficient_attention(
            attention_op=MemoryEfficientAttentionFlashAttentionOp
        )

        # Freeze modules
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.vae.requires_grad_(False)

        # Make only certain upsampler modules trainable
        trainable_modules = [
            'feature_upsampler0',
            'feature_upsampler1',
            'feature_upsampler2',
            'feature_upsampler3'
        ]
        if opt.pretrained_2x_path is not None:
            state_dict = torch.load(to_absolute_path(opt.pretrained_2x_path), map_location='cpu')['state_dict']
            filtered_state_dict = {
                k[len("unet."):]: v
                for k, v in state_dict.items() if any(module in k for module in trainable_modules)
            }
            self.unet.load_state_dict(filtered_state_dict, strict=False)

        for name, param in self.unet.named_parameters():
            if any(module in name for module in trainable_modules):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def setup(self, stage=None):
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            output_type="latent",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        ).to(self.device)

    def training_step(self, batch, batch_idx):
        # Move tensors to the correct device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        images_2x = F.interpolate(batch["images"], scale_factor=0.5, mode="bilinear", align_corners=False)

        with torch.no_grad():
            lr_fea = self.vae.encode(images_2x).latent_dist.sample()
            lr_fea = lr_fea * self.vae.config.scaling_factor

            latents = self.vae.encode(batch["images"]).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        with torch.no_grad():
            encoder_output = self.text_encoder(batch['text_input_ids'], output_hidden_states=True)
            text_embeds = encoder_output.hidden_states[-2]
            encoder_output_2 = self.text_encoder_2(batch['text_input_ids_2'], output_hidden_states=True)
            pooled_text_embeds = encoder_output_2[0]
            text_embeds_2 = encoder_output_2.hidden_states[-2]
            text_embeds = torch.concat([text_embeds, text_embeds_2], dim=-1)

        # Prepare additional conditioning information
        add_time_ids = torch.cat([
            batch["original_size"],
            batch["crop_coords_top_left"],
            batch["target_size"],
        ], dim=1)
        unet_added_cond_kwargs = {"text_embeds": pooled_text_embeds, "time_ids": add_time_ids}

        lr_fea = lr_fea * self.vae.config.scaling_factor
        noise_pred = self.unet(
            sample=noisy_latents,
            lr_fea=lr_fea,
            timestep=timesteps,
            encoder_hidden_states=text_embeds,
            added_cond_kwargs=unet_added_cond_kwargs
        ).sample

        loss = F.mse_loss(noise_pred, noise, reduction="mean")
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        params_to_optimize = list(self.unet.feature_upsampler0.parameters()) + \
                             list(self.unet.feature_upsampler1.parameters()) + \
                             list(self.unet.feature_upsampler2.parameters()) + \
                             list(self.unet.feature_upsampler3.parameters())
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=self.opt.learning_rate,
            weight_decay=self.opt.weight_decay
        )
        return optimizer


class SelfCascadeDataModule(pl.LightningDataModule):
    def __init__(self, opt: DictConfig):
        super().__init__()
        self.opt = opt
        self.tokenizer = CLIPTokenizer.from_pretrained(
            opt.pretrained_model_name_or_path, subfolder="tokenizer"
        )
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            opt.pretrained_model_name_or_path, subfolder="tokenizer_2"
        )
        print("inside DataModule init")

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = MyDataset(
                json_file=to_absolute_path(self.opt.data_json_file),
                tokenizer=self.tokenizer,
                tokenizer_2=self.tokenizer_2,
                size=self.opt.resolution,
                image_root_path=to_absolute_path(self.opt.data_root_path)
            )
        print("inside DataModule setup")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.opt.train_batch_size,
            shuffle=True,
            num_workers=self.opt.dataloader_num_workers,
            collate_fn=collate_fn
        )


@hydra.main(config_path="config", config_name="config_2nodes")
def main(opt: DictConfig):
    print("Configuration:\n", OmegaConf.to_yaml(opt))

    # Define the base directory for results
    base_results_dir = opt.base_results_dir
    subproject_name = opt.get("subproject_name", "default_project")
    run_name = opt.get("run_name", "default_run")
    output_dir = os.path.join(base_results_dir, subproject_name, run_name)
    os.makedirs(output_dir, exist_ok=True)

    model = SelfCascadeModel(opt)
    print("model initialized")
    data_module = SelfCascadeDataModule(opt)
    print("data module initialized")

    # Use WandB logger (or TensorBoard if you prefer)
    logger = WandbLogger(
        project="SelfCascade-SDXL-" + subproject_name,
        name=run_name,
        save_dir=to_absolute_path(output_dir)
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=to_absolute_path(output_dir),
        filename='checkpoint-{step}',
        save_top_k=-1,
        monitor='step',
        mode='max',
        every_n_train_steps=opt.save_n_steps,
        save_on_train_epoch_end=False,
        save_last=True,
        save_weights_only=False
    )

    # Create the time-based checkpoint callback.
    time_checkpoint_callback = TimeCheckpointCallback(
        checkpoint_path=os.path.join(to_absolute_path(output_dir), "temp.ckpt"),
        run_duration=float(opt.auto_save_duration) * 3600
    )

    # Select the desired strategy
    if opt.strategy == "ddp":
        strategy = DDPStrategy(gradient_as_bucket_view=True, find_unused_parameters=True)
    elif opt.strategy == "fsdp":
        custom_wrap_policy = {Attention, Transformer2DModel, DualTransformer2DModel}
        strategy = FSDPStrategy(
            cpu_offload=False,
            auto_wrap_policy=custom_wrap_policy,
            activation_checkpointing_policy={Attention, Transformer2DModel, DualTransformer2DModel},
            sharding_strategy="FULL_SHARD",
        )
    else:
        strategy = None

    callbacks = [checkpoint_callback, time_checkpoint_callback]
    if opt.use_ema:
        ema_callback = EMACallback(model_key="unet", decay=0.9999)
        callbacks.append(ema_callback)

    trainer = pl.Trainer(
        max_steps=opt.max_steps,
        logger=logger,
        log_every_n_steps=5,
        callbacks=callbacks,
        precision='bf16',
        accelerator='gpu',
        devices=1,
        num_nodes=opt.num_nodes,
        strategy=strategy,
        accumulate_grad_batches=opt.gradient_accumulation_steps
    )

    trainer.fit(model, data_module, ckpt_path=opt.resume_checkpoint_path)


if __name__ == "__main__":
    main()
