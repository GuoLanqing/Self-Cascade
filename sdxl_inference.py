import torch
from PIL import Image
from source_override.unet2dconditionalmodel import UNet2DConditionModel
from source_override.pipeline_sdxl_changed import StableDiffusionXLPipeline as TrainedPipeline


class SelfCascadeInference:
    def __init__(self, pretrained_model_path, trained_checkpoint_path, gpu_ids):
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{gpu_ids}")
            print(f"Using GPU: {torch.cuda.get_device_name(self.device)}")
        else:
            self.device = torch.device("cpu")
            print("CUDA is not available. Using CPU.")

        # Load the base pipeline
        self.base_pipeline = TrainedPipeline.from_pretrained(
            pretrained_model_path,
            output_type="latent",
            torch_dtype=torch.float32,
            # torch_dtype=torch.float16,
            # variant="fp16",
            use_safetensors=True
        ).to(self.device)

        # Load the trained pipeline
        self.trained_pipeline = TrainedPipeline.from_pretrained(
            pretrained_model_path,
            torch_dtype=torch.float32,
            # variant="fp16",
            use_safetensors=True
        ).to(self.device)

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
        state_dict = torch.load(trained_checkpoint_path, map_location=self.device)['state_dict']
        # Filter the state dict to include only the trainable modules
        # "unet.feature_upsampler0.0.in_layers.0.weight" to "feature_upsampler0.0.in_layers.0.weight"
        filtered_state_dict = {k[len("unet."):]: v for k, v in state_dict.items() if any(module in k for module in trainable_modules)}

        # Load the filtered state dict
        trained_unet.load_state_dict(filtered_state_dict, strict=False)

        # Replace the trained pipeline's UNet with the trained one
        self.trained_pipeline.unet = trained_unet.to(self.device)

    def generate_image(self, prompt, negative_prompt="", num_inference_steps=50, guidance_scale=7.5, seed=None,
                       strength=0.7, height=2048, width=2048):
        if seed is not None:
            torch.manual_seed(seed)
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        # Generate base image and latents
        # default size is 1024
        base_image, latents = self.base_pipeline(
            prompt=prompt,
            tunning_status=False,  # default is false
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        )

        # upscale_tunningFree_image, _ = self.base_pipeline(
        #     prompt=prompt,
        #     negative_prompt=negative_prompt,
        #     lr_fea=latents,
        #     strength=0.7,
        #     height=2048,
        #     width=2048,
        #     tunning_status=False,
        # )

        # Upscale using trained pipeline
        # upsample to 2048
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

        return base_image[0], upscale_image[0], upscale_image[0]


def save_image(image, filename):
    image.save(filename)


def main():
    pretrained_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
    trained_checkpoint_path = '/home/xinrui/hdd1/xinrui/SelfCascade/results/pexels_txtilm7b_resized_v2_finetune+1k/upsampler_only.pth'
    gpu_ids = 3
    inference = SelfCascadeInference(pretrained_model_path, trained_checkpoint_path, gpu_ids)

    prompt = "Envision a portrait of an elderly woman, her face a canvas of time, framed by a headscarf with muted tones of rust and cream. Her eyes, blue like faded denim. Her attire, simple yet dignified."
    # prompt = "Big fringe scarf with pink tied sweater and pink beanie! Perfect girly fall outfit! Find details on fashion blog daily dose of charm by lauren lindmark"
    negative_prompt = "blurry, ugly, duplicate, poorly drawn, deformed, mosaic"

    base_image, upscale_image, tunningFree_image = inference.generate_image(prompt, negative_prompt=negative_prompt, seed=3407)

    save_image(base_image[0], "results/base_image_elder_16.png")
    save_image(upscale_image[0], "results/upscale_image_elder_16.png")
    print("Images generated and saved as 'base_image_v3_boy.png' and 'upscale_image_v3_boy.png', and 'tunning_free_image_boy.png'")


if __name__ == "__main__":
    main()