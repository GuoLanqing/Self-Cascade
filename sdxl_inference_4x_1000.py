import os
import torch
import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from source_override.unet2dconditionalmodel_upsample import UNet2DConditionModel_upsample
from source_override.unet2dconditionalmodel import UNet2DConditionModel
from source_override.pipeline_sdxl_changed import StableDiffusionXLPipeline as TrainedPipeline
import argparse
from diffusers import AutoencoderKL


class SelfCascadeInference(pl.LightningModule):
    def __init__(self, pretrained_model_path, trained_checkpoint_path_2x, trained_checkpoint_path_4x, output_dir):
        super().__init__()
        self.save_hyperparameters()
        self.output_dir = output_dir

        vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae").to(self.device)
        vae.enable_tiling()

        # Load the base pipeline
        self.base_pipeline = TrainedPipeline.from_pretrained(
            pretrained_model_path,
            output_type="latent",
            torch_dtype=torch.float32,
            use_safetensors=True
        )

        # Load the trained pipeline
        self.trained_pipeline_2x = TrainedPipeline.from_pretrained(
            pretrained_model_path,
            torch_dtype=torch.float32,
            # variant="fp16",
            use_safetensors=True
        ).to(self.device)

        self.trained_pipeline_4x = TrainedPipeline.from_pretrained(
            pretrained_model_path,
            torch_dtype=torch.float32,
            # variant="fp16",
            use_safetensors=True
        ).to(self.device)

        # Load the trained UNet
        trained_unet_2x = UNet2DConditionModel_upsample.from_pretrained(
            pretrained_model_path,
            subfolder="unet",
            low_cpu_mem_usage=False,
            device_map=None
        )
        trained_unet_4x = UNet2DConditionModel_upsample.from_pretrained(
            pretrained_model_path,
            subfolder="unet",
            low_cpu_mem_usage=False,
            device_map=None
        )

        trainable_modules = [
            'feature_upsampler0',
            'feature_upsampler1',
            'feature_upsampler2',
            'feature_upsampler3'
        ]

        # load the trained checkpoint for 2x
        state_dict = torch.load(trained_checkpoint_path_2x, map_location="cpu")['state_dict']
        filtered_state_dict = {k[len("unet."):]: v for k, v in state_dict.items() if
                               any(module in k for module in trainable_modules)}
        # print(filtered_state_dict)
        trained_unet_2x.load_state_dict(filtered_state_dict, strict=False)
        self.trained_pipeline_2x.unet = trained_unet_2x.to(self.device)

        # load the trained checkpoint for 4x
        state_dict = torch.load(trained_checkpoint_path_4x, map_location="cpu")['state_dict']
        filtered_state_dict = {k[len("unet."):]: v for k, v in state_dict.items() if
                               any(module in k for module in trainable_modules)}
        for key in filtered_state_dict.keys():
            print(key)
        trained_unet_4x.load_state_dict(filtered_state_dict, strict=False)
        self.trained_pipeline_4x.unet = trained_unet_4x.to(self.device)

        self.base_pipeline.vae = vae
        self.trained_pipeline_2x.vae = vae
        self.trained_pipeline_4x.vae = vae

    def generate_image(self, prompt, negative_prompt="", num_inference_steps=50, guidance_scale=7.5, seed=None,
                           strength=0.6, height=4096, width=4096):

        self.base_pipeline = self.base_pipeline.to(self.device)
        self.trained_pipeline_2x = self.trained_pipeline_2x.to(self.device)
        self.trained_pipeline_4x = self.trained_pipeline_4x.to(self.device)

        if seed is not None:
            torch.manual_seed(seed)
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        # Generate base image and latents
        base_image, latents = self.base_pipeline(
            prompt=prompt,
            tunning_status=False,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            height=height / 4,
            width=width / 4,
            guidance_scale=guidance_scale,
            generator=generator
        )

        # Upscale using trained pipeline
        # upsample to 2048
        upscale_image_2x, lr_fea_2x = self.trained_pipeline_2x(
            prompt=prompt,
            negative_prompt=negative_prompt,
            lr_fea=latents,
            strength=strength,
            height=height / 2,
            width=width / 2,
            tunning_status=True,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        )

        upscale_image_4x, _ = self.trained_pipeline_4x(
            prompt=prompt,
            negative_prompt=negative_prompt,
            lr_fea=lr_fea_2x,
            strength=strength,
            height=4096,
            width=4096,
            tunning_status=True,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        )

        return base_image[0], upscale_image_2x[0], upscale_image_4x[0]

    def forward(self, batch_prompt):

        prompt = batch_prompt['prompt'][0]  # Assuming batch size of 1
        negative_prompt = batch_prompt['negative_prompt'][0]
        filename = batch_prompt['filename'][0]

        base_image, upscale_image_2x, upscale_image_4x = self.generate_image(prompt, negative_prompt, seed=3407)

        # Save the images
        base_name = os.path.splitext(filename)[0]
        self.save_image(base_image[0], os.path.join(self.output_dir, f"{base_name}_base.png"))
        self.save_image(upscale_image_2x[0], os.path.join(self.output_dir, f"{base_name}_upscale.png"))
        self.save_image(upscale_image_4x[0], os.path.join(self.output_dir, f"{base_name}_upscale_4x.png"))

        # print(f"Images generated and saved for prompt: {filename}")

        # return base_image, upscale_image

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    @staticmethod
    def save_image(image, filename):
        # Convert from tensor to PIL Image and save
        # Image.fromarray((image.cpu().numpy().transpose(1, 2, 0) * 255).astype('uint8')).save(filename)
        image.save(filename)

class PromptDataset(Dataset):
    def __init__(self, prompt_folder, file_list_path, negative_prompt=""):
        with open(file_list_path, 'r') as f:
            self.prompt_files = [line.strip() for line in f.readlines()]
        self.prompt_folder = prompt_folder
        self.negative_prompt = negative_prompt

    def __len__(self):
        return len(self.prompt_files)

    def __getitem__(self, idx):
        prompt_file = self.prompt_files[idx]
        with open(os.path.join(self.prompt_folder, prompt_file), 'r') as f:
            prompt = f.read().strip()
        return {'prompt': prompt, 'negative_prompt': self.negative_prompt, 'filename': prompt_file}


def save_image(image, filename):
    image.save(filename)

def main():
    parser = argparse.ArgumentParser(description="Inference script for SelfCascade model.")

    parser.add_argument('--pretrained_model_path', type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument('--trained_checkpoint_path_2x', type=str, required=True,
                        help='Path to trained checkpoint for 2x model')
    parser.add_argument('--trained_checkpoint_path_4x', type=str, required=True,
                        help='Path to trained checkpoint for 4x model')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='Folder to store output results')
    parser.add_argument('--prompt_folder', type=str, required=True,
                        help='Folder containing prompt captions')
    parser.add_argument('--file_list_path', type=str, required=True,
                        help='Path to file list for testing prompts')
    parser.add_argument('--negative_prompt', type=str,
                        default="blurry, ugly, duplicate, poorly drawn, deformed, mosaic")
    parser.add_argument('--gpu_devices', type=int, nargs='+', default=[0])

    args = parser.parse_args()

    # Initialize the model
    model = SelfCascadeInference(args.pretrained_model_path, args.trained_checkpoint_path_2x, args.trained_checkpoint_path_4x, args.output_folder)

    negative_prompt = "blurry, ugly, duplicate, poorly drawn, deformed, mosaic"

    # Create a dataset and dataloader
    dataset = PromptDataset(args.prompt_folder, args.file_list_path, args.negative_prompt)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4)

    # Initialize the trainer
    trainer = pl.Trainer(accelerator='gpu', devices=args.gpu_devices, strategy='ddp', precision='bf16')

    # Run the model
    trainer.predict(model, dataloaders=dataloader)


if __name__ == "__main__":
    main()
