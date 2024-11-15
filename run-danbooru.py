import torch
import argparse
from diffusers import StableDiffusionPipeline
from diffusers.models import AutoencoderKL
from safetensors.torch import load_file
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
parser.add_argument('--target', dest='target', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--wide', dest='wide', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--tall', dest='tall', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--prompt', dest='prompt')
parser.add_argument('--lora', dest='lora')
parser.add_argument('--autoencoder', dest='autoencoder')

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
    if args.autoencoder is not None:
        folder = slugify(args.autoencoder)
    else:
        folder = slugify(model_id)
    folder = 'danbooru-'+folder

    if args.autoencoder is not None:
        vae = AutoencoderKL.from_pretrained(args.autoencoder)
        pipe = StableDiffusionPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.float16)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    
    if args.lora is not None:
        folder += "-" + slugify(args.lora)
        load_lora_weights(pipe, args.lora)
    
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
        height=512
        width=512

    pipe = pipe.to(device)
    
    if args.target:
        targets = []
        for fname in os.listdir(folder):
            if not os.path.isdir(os.path.join(folder,fname)):
                targets.append(fname.split("-")[2])
        
        targets.reverse()
        tfolder = folder
    elif args.prompt is not None:
        targets = [str(pipe.tokenizer.encode(args.prompt)[1:-1])]
        tfolder = folder
    else:
        targets = [None]
    for target in targets:
        if target is not None:
            folder = os.path.join(tfolder,target)
        folder+="american_flag_shirt midriff prepend"
        if os.path.exists(folder):
            print(f"{folder} already exists")
            continue
        print(folder)
        os.mkdir(folder)
        loss = torch.nn.MSELoss()
        master_prompts = []
        import csv
        with open('danbooru-tags.csv', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0] != "":
                    master_prompts.append("american_flag_shirt, midriff, "+row[0]+",")
        print("warning: Some danbooru tags are very NSFW, make sure your filters work.")

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
                        if target is None:
                            prompt2Latent.append((prompts[i],latents[i].clone()))
                        else:
                            prompt2Latent.append((prompts[0],latents[i].clone()))
                return None
            rets.append(ret)
            return ret

        try: step
        except NameError: step = None
        if step is None:
            step = pipe.scheduler.__class__.step
        pipe.scheduler.__class__.step = test
        
        
        torch.cuda.empty_cache()
        minLoss = 0.01
        depth = 5
        import random
        se = 418947 # random.randint(0,999999) #mostly just to allow us all to work on the same images
        print()
        print(se)
        prompt2Latent = []
        for p in range(len(master_prompts)-1,0,-1*num_images):
            if p < num_images:
                prompts = master_prompts[p::-1]
            else:
                prompts = master_prompts[p:p-num_images:-1]
            torch.manual_seed(se)
            attempts = 5
            imagesPerPrompt = 1
            if target is not None:
                import json
                prompt_pids = [json.loads(target)]
                prompts = pipe.tokenizer.batch_decode(prompt_pids)
                attempts = 20
                imagesPerPrompt = num_images
            for j in range(attempts):
                rets = []
                shape = (num_images, pipe.unet.config.in_channels, height // pipe.vae_scale_factor, width // pipe.vae_scale_factor)
                latents = torch.randn(shape, generator=None, device=device, dtype=pipe.text_encoder.dtype, layout=torch.strided).to(device)
                try:
                    imagesnp = pipe(prompts, latents=latents, num_images_per_prompt=imagesPerPrompt, num_inference_steps=50, output_type="np.array").images
                except AttributeError:
                    pass
            if len(prompt2Latent) >= num_images or p<=(num_images*2) or target is not None:
                skip = True
                tprompts = []
                for j in range(len(prompt2Latent)):
                    tprompt,latent = prompt2Latent[j]
                    latents[len(tprompts)] = latent
                    tprompts.append(tprompt)
                    if len(tprompts)<num_images and j<len(prompt2Latent)-1:
                        continue
                    tprompt_pids = [pipe.tokenizer.encode(proms)[1:-1] for proms in tprompts] #the [1:-1] are to remove the START and END tokens
                    rets = []
                    imagesnp = pipe(tprompts, latents=latents[:len(tprompts)], num_images_per_prompt=1, num_inference_steps=50, output_type="np.array").images
                    imagesnpSafe,has_nsfw_concept = safety_checker(np.copy(imagesnp),device,pipe.text_encoder.dtype)
                    images = pipe.numpy_to_pil(imagesnp)
                    
                    for i in range(len(images)):
                        im = images[i]
                        l = loss(rets[0].prev_sample[i],rets[depth].prev_sample[i])
                        if len(tprompt_pids[i]) > 10:
                            tprompt_pids[i] = tprompt_pids[i][:10]
                            tprompt_pids[i].append("...")
                        fname = os.path.join(folder,f'{l:.05}-{slugify(tprompts[i])}-{tprompt_pids[i]}-{i}.png')
                        if np.any(imagesnpSafe[i] != imagesnp[i]):
                            fname+=".nsfw"
                        im.save(fname,'PNG')

                    tprompts = []
                    tprompt_pids = []
                skip = False
                prompt2Latent = []
            if target is not None:
                break
print(prompt2Latent)
print(time.time()-start)