import torch
import sys
from diffusers import StableDiffusionPipeline
import unicodedata
from tqdm.auto import tqdm
import re
import time
import os
import numpy as np
start = time.time()
num_images=16

try:
    model_id = sys.argv[1]
except:
    print('python run.py MODEL_ID \nex. python combine.py "CompVis/stable-diffusion-v1-1"')
    exit()

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

baseFolder = slugify(model_id)
for target in os.listdir(baseFolder):
    if not os.path.isdir(os.path.join(baseFolder,target)):
        continue
    folder = os.path.join(baseFolder,target)
    print(target)
    ims = []
    import os
    from PIL import Image
    import torchvision.transforms as transforms
    transform = transforms.ToTensor()
    for fname in os.listdir(folder):
        fname = os.path.join(folder,fname)
        ims.append(transform(Image.open(fname)))
    fin = ims[0]
    for im in ims[1:]:
        fin+=im
    fin /=len(ims)
    finIm = transforms.ToPILImage()(fin)
    finIm.save(os.path.join(folder,'combined.png'))
