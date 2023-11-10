import hashlib
import json
import os
import shutil
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from weights import WeightsDownloadCache

import numpy as np
import torch
from cog import BasePredictor, Input, Path
from diffusers import DiffusionPipeline, LCMScheduler
from diffusers.models.attention_processor import LoRAAttnProcessor2_0
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from diffusers.utils import load_image
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import CLIPImageProcessor

from dataset_and_utils import TokenEmbeddingsHandler

SDXL_MODEL_CACHE = "./sdxl-cache"
SAFETY_CACHE = "./safety-cache"
FEATURE_EXTRACTOR = "./feature-extractor"
SDXL_URL = "https://weights.replicate.delivery/default/sdxl/sdxl-vae-upcast-fix.tar"
REFINER_URL = (
    "https://weights.replicate.delivery/default/sdxl/refiner-no-vae-no-encoder-1.0.tar"
)
SAFETY_URL = "https://weights.replicate.delivery/default/sdxl/safety-1.0.tar"
lcm_lora_id = "latent-consistency/lcm-lora-sdxl"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def load_trained_weights(self, weights, pipe):
        from no_init import no_init_or_tensor

        # weights can be a URLPath, which behaves in unexpected ways
        weights = str(weights)
        if self.tuned_weights == weights:
            print("skipping loading .. weights already loaded")
            return

        self.tuned_weights = weights
        local_weights_cache = self.weights_cache.ensure(weights)

        # load UNET
        print("Loading fine-tuned model")
        self.is_lora = False

        maybe_unet_path = os.path.join(local_weights_cache, "unet.safetensors")
        if not os.path.exists(maybe_unet_path):
            print("Does not have Unet. assume we are using LoRA")
            self.is_lora = True

        if not self.is_lora:
            print("Loading Unet")

            new_unet_params = load_file(
                os.path.join(local_weights_cache, "unet.safetensors")
            )
            # this should return _IncompatibleKeys(missing_keys=[...], unexpected_keys=[])
            pipe.unet.load_state_dict(new_unet_params, strict=False)

        else:
            print("Loading Unet LoRA")

            unet = pipe.unet

            tensors = load_file(os.path.join(local_weights_cache, "lora.safetensors"))

            unet_lora_attn_procs = {}
            name_rank_map = {}
            for tk, tv in tensors.items():
                # up is N, d
                if tk.endswith("up.weight"):
                    proc_name = ".".join(tk.split(".")[:-3])
                    r = tv.shape[1]
                    name_rank_map[proc_name] = r

            for name, attn_processor in unet.attn_processors.items():
                cross_attention_dim = (
                    None
                    if name.endswith("attn1.processor")
                    else unet.config.cross_attention_dim
                )
                if name.startswith("mid_block"):
                    hidden_size = unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(unet.config.block_out_channels))[
                        block_id
                    ]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = unet.config.block_out_channels[block_id]
                with no_init_or_tensor():
                    module = LoRAAttnProcessor2_0(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        rank=name_rank_map[name],
                    )
                unet_lora_attn_procs[name] = module.to("cuda", non_blocking=True)

            unet.set_attn_processor(unet_lora_attn_procs)
            unet.load_state_dict(tensors, strict=False)

        # load text
        handler = TokenEmbeddingsHandler(
            [pipe.text_encoder, pipe.text_encoder_2], [pipe.tokenizer, pipe.tokenizer_2]
        )
        handler.load_embeddings(os.path.join(local_weights_cache, "embeddings.pti"))

        # load params
        with open(os.path.join(local_weights_cache, "special_params.json"), "r") as f:
            params = json.load(f)
        self.token_map = params

        self.tuned_model = True

    def setup(self, weights: Optional[Path] = None):
        """Load the model into memory to make running multiple predictions efficient"""
        start = time.time()
        self.tuned_model = False
        self.tuned_weights = None
        if str(weights) == "weights":
            weights = None

        self.weights_cache = WeightsDownloadCache()

        print("Loading safety checker...")
        if not os.path.exists(SAFETY_CACHE):
            download_weights(SAFETY_URL, SAFETY_CACHE)
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            SAFETY_CACHE, torch_dtype=torch.float16
        ).to("cuda")
        self.feature_extractor = CLIPImageProcessor.from_pretrained(FEATURE_EXTRACTOR)

        if not os.path.exists(SDXL_MODEL_CACHE):
            download_weights(SDXL_URL, SDXL_MODEL_CACHE)

        print("Loading sdxl txt2img pipeline...")
        self.txt2img_pipe = DiffusionPipeline.from_pretrained(
            SDXL_MODEL_CACHE,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        self.txt2img_pipe.load_lora_weights(lcm_lora_id)
        self.txt2img_pipe.scheduler = LCMScheduler.from_config(self.txt2img_pipe.scheduler.config)
        self.txt2img_pipe.to("cuda")

        self.is_lora = False
        if weights or os.path.exists("./trained-model"):
            self.load_trained_weights(weights, self.txt2img_pipe)

        print("setup took: ", time.time() - start)

    def load_image(self, path):
        shutil.copyfile(path, "/tmp/image.png")
        return load_image("/tmp/image.png").convert("RGB")

    def run_safety_checker(self, image):
        safety_checker_input = self.feature_extractor(image, return_tensors="pt").to(
            "cuda"
        )
        np_image = [np.array(val) for val in image]
        image, has_nsfw_concept = self.safety_checker(
            images=np_image,
            clip_input=safety_checker_input.pixel_values.to(torch.float16),
        )
        return image, has_nsfw_concept

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="An astronaut riding a rainbow unicorn",
        ),
        negative_prompt: str = Input(
            description="Input Negative Prompt",
            default="",
        ),
        image: Path = Input(
            description="Input image for img2img or inpaint mode",
            default=None,
        ),
        mask: Path = Input(
            description="Input mask for inpaint mode. Black areas will be preserved, white areas will be inpainted.",
            default=None,
        ),
        width: int = Input(
            description="Width of output image",
            default=1024,
        ),
        height: int = Input(
            description="Height of output image",
            default=1024,
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=10, default=8
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=50, default=2.0
        ),
        prompt_strength: float = Input(
            description="Prompt strength when using img2img / inpaint. 1.0 corresponds to full destruction of information in image",
            ge=0.0,
            le=1.0,
            default=0.8,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        apply_watermark: bool = Input(
            description="Applies a watermark to enable determining if an image is generated in downstream applications. If you have other provisions for generating or deploying images safely, you can use this to disable watermarking.",
            default=True,
        ),
        lora_scale: float = Input(
            description="LoRA additive scale. Only applicable on trained models.",
            ge=0.0,
            le=1.0,
            default=0.6,
        ),
        replicate_weights: str = Input(
            description="Replicate LoRA weights to use. Leave blank to use the default weights.",
            default=None,
        ),
        disable_safety_checker: bool = Input(
            description="Disable safety checker for generated images. This feature is only available through the API. See [https://replicate.com/docs/how-does-replicate-work#safety](https://replicate.com/docs/how-does-replicate-work#safety)",
            default=False
        )
    ) -> List[Path]:
        """Run a single prediction on the model."""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if replicate_weights:
            self.load_trained_weights(replicate_weights, self.txt2img_pipe)
        
        # OOMs can leave vae in bad state
        if self.txt2img_pipe.vae.dtype == torch.float32:
            self.txt2img_pipe.vae.to(dtype=torch.float16)

        sdxl_kwargs = {}
        if self.tuned_model:
            # consistency with fine-tuning API
            for k, v in self.token_map.items():
                prompt = prompt.replace(k, v)
        print(f"Prompt: {prompt}")
        if image and mask:
            print("inpainting mode")
            sdxl_kwargs["image"] = self.load_image(image)
            sdxl_kwargs["mask_image"] = self.load_image(mask)
            sdxl_kwargs["strength"] = prompt_strength
            sdxl_kwargs["width"] = width
            sdxl_kwargs["height"] = height
            pipe = self.inpaint_pipe
        elif image:
            print("img2img mode")
            sdxl_kwargs["image"] = self.load_image(image)
            sdxl_kwargs["strength"] = prompt_strength
            pipe = self.img2img_pipe
        else:
            print("txt2img mode")
            sdxl_kwargs["width"] = width
            sdxl_kwargs["height"] = height
            pipe = self.txt2img_pipe

        if not apply_watermark:
            # toggles watermark for this prediction
            watermark_cache = pipe.watermark
            pipe.watermark = None
            self.refiner.watermark = None

        generator = torch.Generator("cuda").manual_seed(seed)

        common_args = {
            "prompt": [prompt] * num_outputs,
            "negative_prompt": [negative_prompt] * num_outputs,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
        }

        if self.is_lora:
            sdxl_kwargs["cross_attention_kwargs"] = {"scale": lora_scale}

        output = pipe(**common_args, **sdxl_kwargs)

        if not apply_watermark:
            pipe.watermark = watermark_cache
            self.refiner.watermark = watermark_cache

        if not disable_safety_checker:
            _, has_nsfw_content = self.run_safety_checker(output.images)

        output_paths = []
        for i, image in enumerate(output.images):
            if not disable_safety_checker:
                if has_nsfw_content[i]:
                    print(f"NSFW content detected in image {i}")
                    continue
            output_path = f"/tmp/out-{i}.png"
            image.save(output_path)
            output_paths.append(Path(output_path))

        if len(output_paths) == 0:
            raise Exception(
                f"NSFW content detected. Try running it again, or try a different prompt."
            )

        return output_paths
