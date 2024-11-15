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
num_images=16

parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
parser.add_argument('model_id')
parser.add_argument('--device', dest='device', default='cuda')
parser.add_argument('--target', dest='target')
parser.add_argument('--autoencoder', dest='autoencoder')

args = parser.parse_args()
model_id = args.model_id
device = args.device

def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')

with torch.no_grad():
    if args.autoencoder is not None:
        folder = slugify(args.autoencoder)
    else:
        folder = slugify(model_id)
    if args.target is not None:
        folder = os.path.join(folder,args.target)
    if os.path.exists(folder):
        print("folder already exists")
        exit()
    os.mkdir(folder)
    loss = torch.nn.MSELoss()
    if args.autoencoder is not None:
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
        pipe = StableDiffusionPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.float16)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)
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
        pbar.set_description_str(str(len(prompt2Latent))+" "+str(p//num_images)+" "+str(j)+" "+str(maxL.item()))
        return pbar
    pipe.progress_bar = progress

    skip = False
    def test(*funcArgs, **kwargs):
        global maxL
        ret = step(*funcArgs, **kwargs)
        if len(rets) == depth and not skip:
            maxL = 0
            for i in range(num_images):
                l = loss(rets[0].prev_sample[i],ret.prev_sample[i])
                maxL = max(maxL, l)
                if l > minLoss:
                    if args.target is None:
                        prompt2Latent.append((prompt_pids[i],latents[i].clone()))
                    else:
                        prompt2Latent.append((prompt_pids[0],latents[i].clone()))
            return None
        rets.append(ret)
        return ret

    try: step
    except NameError: step = None
    if step is None:
        step = pipe.scheduler.__class__.step
    pipe.scheduler.__class__.step = test
    
    
    torch.cuda.empty_cache()
    minLoss = 0.005
    depth = 5
    import random
    se = 418947 # random.randint(0,999999) #mostly just to allow us all to work on the same images
    print()
    print(se)
    prompt2Latent = []
    for p in range(len(pipe.tokenizer.get_vocab().keys())-1,0,-1*num_images):
        prompt_pids = [ [pId] for pId in range(p,p-num_images,-1) ]
        prompts = pipe.tokenizer.batch_decode(prompt_pids)
        torch.manual_seed(se)
        attempts = 5
        imagesPerPrompt = 1
        if args.target is not None:
            import json
            prompt_pids = [json.loads(args.target)]
            prompts = pipe.tokenizer.batch_decode(prompt_pids)
            attempts = 20
            imagesPerPrompt = num_images
        for j in range(attempts):
            rets = []
            shape = (num_images, pipe.unet.config.in_channels, pipe.unet.config.sample_size, pipe.unet.config.sample_size)
            latents = torch.randn(shape, generator=None, device=device, dtype=pipe.text_encoder.dtype, layout=torch.strided).to(device)
            try:
                imagesnp = pipe(prompts, latents=latents, num_images_per_prompt=imagesPerPrompt, num_inference_steps=50, output_type="np.array").images
            except AttributeError:
                pass
        if len(prompt2Latent) >= num_images or p<=(num_images*2) or args.target is not None:
            skip = True
            tprompt_pids = []
            for j in range(len(prompt2Latent)):
                prompt_pid,latent = prompt2Latent[j]
                latents[len(tprompt_pids)] = latent
                tprompt_pids.append(prompt_pid)
                if len(tprompt_pids)<num_images and j<len(prompt2Latent)-1:
                    continue
                tprompts = pipe.tokenizer.batch_decode(tprompt_pids)
                rets = []
                imagesnp = pipe(tprompts, latents=latents[:len(tprompts)], num_images_per_prompt=1, num_inference_steps=50, output_type="np.array").images
                imagesnpSafe,has_nsfw_concept = safety_checker(np.copy(imagesnp),device,pipe.text_encoder.dtype)
                images = pipe.numpy_to_pil(imagesnp)
                
                for i in range(len(images)):
                    im = images[i]
                    l = loss(rets[0].prev_sample[i],rets[depth].prev_sample[i])
                    fname = os.path.join(folder,f'{l:.05}-{slugify(tprompts[i])}-{tprompt_pids[i]}-{i}.png')
                    if np.any(imagesnpSafe[i] != imagesnp[i]):
                        fname+=".nsfw"
                    im.save(fname,'PNG')

                tprompts = []
                tprompt_pids = []
            skip = False
            prompt2Latent = []
        if args.target is not None:
            break
print(prompt2Latent)
print(time.time()-start)