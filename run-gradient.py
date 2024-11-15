import torch
import argparse
from diffusers import StableDiffusionPipeline
from diffusers.models import AutoencoderKL
import unicodedata
from tqdm.auto import tqdm
import re
import time
import os
import numpy as np
start = time.time()
num_images=10
small = 540
large = 768

parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
parser.add_argument('model_id')
parser.add_argument('--device', dest='device', default='cuda')
parser.add_argument('--wide', dest='wide', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--tall', dest='tall', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--lora', dest='lora')

args = parser.parse_args()
model_id = args.model_id
device = args.device

# source https://github.com/huggingface/diffusers/issues/3064
def load_lora_weights(pipeline, checkpoint_path):
    dtype = pipe.text_encoder.dtype
    pipe.to(device)
    # load base model
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    alpha = 1.0
    # load LoRA weight from .safetensors
    if checkpoint_path.endswith(".safetensors"):
        state_dict = load_file(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path)
    visited = []

    # directly update weight in diffusers model
    for key in state_dict:
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        # as we have set the alpha beforehand, so just skip
        if ".alpha" in key or key in visited:
            continue

        if "text" in key:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = pipeline.unet

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        pair_keys = []
        if "lora_down" in key:
            pair_keys.append(key.replace("lora_down", "lora_up"))
            pair_keys.append(key)
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace("lora_up", "lora_down"))

        # update weight
        if len(state_dict[pair_keys[0]].shape) == 4:
            weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(dtype).to(device)
            weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(dtype).to(device)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
        else:
            weight_up = state_dict[pair_keys[0]].to(dtype).to(device)
            weight_down = state_dict[pair_keys[1]].to(dtype).to(device)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down)

        # update visited list
        for item in pair_keys:
            visited.append(item)

    return pipeline

def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = os.path.basename(value)
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '_', value).strip('-_')

with torch.no_grad():
    folder = slugify(model_id)
    
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    if args.lora is not None:
        folder += "-" + slugify(args.lora)
        load_lora_weights(pipe, args.lora)
    folder+="-gradient"
    
    if args.wide:
        folder += "-wide"
        width=large
        height=small
    elif args.tall:
        folder += "-tall"
        width=small
        height=large
    else:
        folder += "-square"
        height=large
        width=large
    
    if os.path.exists(folder):
        print("folder already exists")
        exit()
    os.mkdir(folder)
    loss = torch.nn.MSELoss()
    def safe(images, device, dtype):
        return images, None
    safety_checker = pipe.run_safety_checker
    pipe.run_safety_checker=safe
    pbar = tqdm(total=50, leave=False)
    maxL = torch.tensor(0)
    def progress(total=None):
        global pbar
        pbar.clear()
        pbar = tqdm(total=50, leave=False)
        return pbar
    pipe.progress_bar = progress

    skip = False
    class test2:
        config=None
        def __init__(self) -> None:
            self.config = unet.config

        def __call__(self,
                    latents,
                    t,
                    encoder_hidden_states=None,
                    cross_attention_kwargs=None,
                    return_dict=False):
            global maxL
            if not skip:
                extra_step_kwargs = {}
                prompt_embeds = torch.randn(encoder_hidden_states.shape,requires_grad=True,dtype=encoder_hidden_states.dtype).to('cuda')
                # encoder_hidden_states == prompt_embeds
                unet.eval()
                pipe.vae.requires_grad_(False)
                pipe.text_encoder.requires_grad_(False)
                pipe.scheduler.timesteps.requires_grad_(False)
                pipe.scheduler.timesteps[0].requires_grad_(False)
                pipe.scheduler.timesteps[1].requires_grad_(False)
                unet.requires_grad_(True)
                latents.requires_grad_(False)
                
                optimizer = torch.optim.Adam([prompt_embeds], lr=0.0001)
                for i in range(10):
                    optimizer.zero_grad()
                    unet.zero_grad()
                    tempLatents = latents.clone().detach()
                    t = pipe.scheduler.timesteps[0]
                    print(pipe.scheduler.timesteps[0])
                    print(pipe.scheduler.timesteps[1])
                    noise_pred = unet(
                        tempLatents, 
                        t,  
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        return_dict=False,
                    )[0]
                    tempLatents = pipe.scheduler.step(
                        noise_pred, 
                        t, 
                        tempLatents, 
                        **extra_step_kwargs, 
                        return_dict=return_dict
                    )[0]
                    t = pipe.scheduler.timesteps[1]
                    noise_pred = unet(
                        tempLatents, 
                        t,  
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        return_dict=False,
                    )[0]
                    temp2Latents = pipe.scheduler.step(
                        noise_pred, 
                        t, 
                        tempLatents, 
                        **extra_step_kwargs, 
                        return_dict=return_dict
                    )[0]
                    l = loss(tempLatents,temp2Latents, requires_grad=True)
                    l.backward()
                    optimizer.step()
                    print(str(l.item()))
                    del(temp2Latents)
                    del(tempLatents)
                    del(l)
                return None

    try: unet
    except NameError: unet = None
    if unet is None:
        unet = pipe.unet
    pipe.unet = test2()

    shape = (num_images, unet.config.in_channels, height // pipe.vae_scale_factor, width // pipe.vae_scale_factor)
    latents = torch.randn(shape, generator=None, device=device, dtype=pipe.text_encoder.dtype, layout=torch.strided).to(device)
    imagesnp = pipe("None", guidance_scale=0, latents=latents, num_images_per_prompt=num_images, num_inference_steps=50, output_type="np.array").images
    exit()    
    torch.cuda.empty_cache()
    minLoss = 0.001
    depth = 5
    import random
    prompt2Latent = []
    for p in range(len(pipe.tokenizer.get_vocab().keys())-1,0,-1*num_images):
        seed = random.randint(0,999999)
        torch.manual_seed(seed)
        rets = []
        shape = (num_images, pipe.unet.config.in_channels, height // pipe.vae_scale_factor, width // pipe.vae_scale_factor)
        latents = torch.randn(shape, generator=None, device=device, dtype=pipe.text_encoder.dtype, layout=torch.strided).to(device)
        try:
            imagesnp = pipe("None", guidance_scale=0, latents=latents, num_images_per_prompt=num_images, num_inference_steps=50, output_type="np.array").images
        except AttributeError:
            pass
        if len(prompt2Latent) >= num_images or p<=(num_images*2):
            skip = True
            seeds = []
            for j in range(len(prompt2Latent)):
                seed,latent = prompt2Latent[j]
                latents[len(seeds)] = latent
                seeds.append(seed)
                if len(seeds)<num_images and j<len(prompt2Latent)-1:
                    continue
                rets = []
                imagesnp = pipe("None", guidance_scale=0, latents=latents[:len(seeds)], num_images_per_prompt=num_images, num_inference_steps=50, output_type="np.array").images
                imagesnpSafe,has_nsfw_concept = safety_checker(np.copy(imagesnp),device,pipe.text_encoder.dtype)
                images = pipe.numpy_to_pil(imagesnp)
                
                for i in range(len(images)):
                    im = images[i]
                    l = loss(rets[0].prev_sample[i],rets[depth].prev_sample[i])
                    fname = os.path.join(folder,f'{l:.05}-{seeds[i]}-{i}.png')
                    if np.any(imagesnpSafe[i] != imagesnp[i]):
                        fname+=".nsfw"
                    im.save(fname,'PNG')
            skip = False
            prompt2Latent = []
print(prompt2Latent)
print(time.time()-start)