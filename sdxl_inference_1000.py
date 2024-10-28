import os
import torch
import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from source_override.unet2dconditionalmodel import UNet2DConditionModel
from source_override.pipeline_sdxl_changed import StableDiffusionXLPipeline as TrainedPipeline
import argparse


class SelfCascadeInference(pl.LightningModule):
    def __init__(self, pretrained_model_path, trained_checkpoint_path, output_dir):
        super().__init__()
        self.save_hyperparameters()
        self.output_dir = output_dir

        # Load the base pipeline
        self.base_pipeline = TrainedPipeline.from_pretrained(
            pretrained_model_path,
            output_type="latent",
            torch_dtype=torch.float32,
            use_safetensors=True
        )

        # Load the trained pipeline
        self.trained_pipeline = TrainedPipeline.from_pretrained(
            pretrained_model_path,
            torch_dtype=torch.float32,
            use_safetensors=True
        )

        # Load the trained UNet
        trained_unet = UNet2DConditionModel.from_pretrained(
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
        state_dict = torch.load(trained_checkpoint_path, map_location='cpu')['state_dict']
        filtered_state_dict = {k[len("unet."):]: v for k, v in state_dict.items() if
                               any(module in k for module in trainable_modules)}

        # Load the filtered state dict
        trained_unet.load_state_dict(filtered_state_dict, strict=False)

        # Replace the trained pipeline's UNet with the trained one
        self.trained_pipeline.unet = trained_unet

    def generate_image(self, prompt, negative_prompt="", num_inference_steps=50, guidance_scale=7.5, seed=None,
                       strength=0.7, height=2048, width=2048):

        self.base_pipeline = self.base_pipeline.to(self.device)
        self.trained_pipeline = self.trained_pipeline.to(self.device)

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
            guidance_scale=guidance_scale,
            generator=generator
        )

        # Upscale using trained pipeline
        upscale_image, _ = self.trained_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            lr_fea=latents,
            strength=strength,
            height=height,
            width=width,
            tunning_status=True,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        )

        return base_image[0], upscale_image[0]

    def forward(self, batch_prompt):

        prompt = batch_prompt['prompt'][0]  # Assuming batch size of 1
        negative_prompt = batch_prompt['negative_prompt'][0]
        filename = batch_prompt['filename'][0]

        base_image, upscale_image = self.generate_image(prompt, negative_prompt, seed=3407)

        # Save the images
        base_name = os.path.splitext(filename)[0]
        self.save_image(base_image[0], os.path.join(self.output_dir, f"{base_name}_base.png"))
        self.save_image(upscale_image[0], os.path.join(self.output_dir, f"{base_name}_upscale.png"))

        # print(f"Images generated and saved for prompt: {filename}")

        return base_image, upscale_image

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
    parser = argparse.ArgumentParser(description="Run inference with specified paths and prompts.")

    # Add arguments
    parser.add_argument('--pretrained_model_path', type=str, default="stabilityai/stable-diffusion-xl-base-1.0",
                        help="Path to the pretrained model.")
    parser.add_argument('--trained_checkpoint_path', type=str, required=True, help="Path to the trained checkpoint.")
    parser.add_argument('--prompt_folder', type=str, required=True, help="Path to the file containing prompts.")
    parser.add_argument('--output_folder', type=str, required=True, help="Path to save the inference images.")
    parser.add_argument('--file_list_path', type=int, required=True, help="txt file path of text file names")

    # Parse arguments
    args = parser.parse_args()


    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    # Initialize the model
    model = SelfCascadeInference(args.pretrained_model_path, args.trained_checkpoint_path, args.output_folder)

    negative_prompt = "blurry, ugly, duplicate, poorly drawn, deformed, mosaic"

    # Create a dataset and dataloader
    dataset = PromptDataset(args.prompt_folder, args.file_list_path, negative_prompt)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4)

    # Initialize the trainer
    trainer = pl.Trainer(accelerator='gpu', devices=[2, 3], strategy='ddp', precision='bf16')

    # Run the model
    trainer.predict(model, dataloaders=dataloader)


if __name__ == "__main__":
    main()