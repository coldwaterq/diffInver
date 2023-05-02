import sys
import os
import markdown
import torch
import torchvision.transforms as transforms
from PIL import Image
loss = torch.nn.MSELoss()
transform = transforms.ToTensor()

maxValue = 0.1
percentMultiplier = 100/maxValue

def compare(fname1, fname2):
    im1 = transform(Image.open(fname1))
    im2 = transform(Image.open(fname2))
    return loss(im1, im2)

models = []
for folder in os.listdir('.'):
    if os.path.isdir(folder) and not folder.startswith('.'):
        models.append(folder)
    if folder.endswith(".html"):
        os.remove(folder)

ids = {}
modelsIds = {}
for model in models:
    for fname in os.listdir(model):
        if os.path.isdir(os.path.join(model,fname)):
            continue
        id = fname.split("-")[2]
        path = os.path.join(model,fname)
        try:
            if model not in ids[id]:
                ids[id].append(model)
        except KeyError:
            ids[id] = [model]
        try:
            modelsIds[model][id].append(path)
        except KeyError:
            try:
                modelsIds[model][id] = [path]
            except KeyError:
                modelsIds[model] = {id:[path]}

modelTotalPaths = {}
for model in models:
    modelTotalPaths[model]=len(os.listdir(model))

for model in models:
    print(model)
    fnames = os.listdir(model)
    fnames.reverse()
    finishedIds = []
    unique = []
    similarity = {}
    similarCount = {}
    shared = []
    for fname in fnames:
        if os.path.isdir(os.path.join(model,fname)):
            continue
        id = fname.split("-")[2]
        if id in finishedIds:
            continue
        name = fname.split("-")[1]
        strength = float(fname.split("-")[0])*percentMultiplier
        if name == '':
            name="????????????"
        
        if len(ids[id]) > 1:
            
            paths = modelsIds[model][id]
            data = [[name,strength,999]]
            fullMin = 999
            for path in paths:
                minScore = 9999
                bestPeer = None
                bestModel = None
                for peerModel in ids[id]:
                    if peerModel == model:
                        continue
                    tempMinScore = 9999
                    for peerPath in modelsIds[peerModel][id]:
                        score = compare(path, peerPath)
                        tempMinScore = min(score,tempMinScore)
                        if score < minScore:
                            minScore = score
                            bestPeer = peerPath
                            bestModel = peerModel
                    try: 
                        similarity[peerModel]+=tempMinScore
                        similarCount[peerModel]+=1
                    except KeyError:
                        similarity[peerModel]=tempMinScore
                        similarCount[peerModel]=1    
                assert(bestPeer is not None and bestModel is not None)
                data.append([minScore, path, bestPeer, bestModel])
                fullMin = min(fullMin,minScore)
            data[0][2] = fullMin
            shared.append(data)
        else:
            if len(unique) == 20:
                continue
            unique.append([[name,strength,id],*modelsIds[model][id]])
        finishedIds.append(id)
    
    document=""
    TOC = f"# {model}\n\n"
    TOC += "## Table of Contents\n"
    # TOC += "- [Model Similarity](#modelSimilarity)\n"
    # document += f'## Model Similarity## {{: #modelSimilarity }}\n'
    # document += '\n\nThis is a comparison of this model to every other model with data collected on it. A high count seems to correlate to number of steps removed from this model. As in Parent vs grandparent in terms of retraining. Average similarity seems to corelate famillies together. **More data is required to draw concrete conclusions.**\n\n'
    
    # document += f'\n\n{"model": <50} | {"count": <5} | {"percentage of self": <30} | {"percentage of peer": <30} | {"average similarity": <30}\n'
    # document += f'{"-"*50} | {"-"*5} | {"-"*30} | {"-"*30} | {"-"*30}\n'
    # for key in similarCount.keys():
    #     document += f"{key: <50} | {similarCount[key]: <5} | {similarCount[key]/modelTotalPaths[model]*100: <30} | {similarCount[key]/modelTotalPaths[key]*100: <30} | {(similarity[key]/similarCount[key]).item(): <30}\n"
    
    TOC += "- [Top 20 Unique Overfit Images(Strength)](#sharedoverfit)\n"
    document += f'\n\n## Unique Overfit Images## {{: #sharedoverfit }}\n'
    document += '\n\nThe higher the strength, the more likely these are reproductions of the training data. >~50% are probably similar to training images. >~10% and <50% probably contain an element of a training image (border, stock photo elements, face). >~5% and <10% are likely reproducable, and that\'s it. These were produced with 5 samples, generating more samples may produce higher scores. In addition to the strength, the number of times the same image appears also indicates how overfit the image is.\n\n'

    for images in unique:
        name,strength,id = images[0]
        images = images[1:]
        images.sort(reverse=True)
        document+=f'### {name}### {{: #{id} }}\n'
        title = f'({strength:0.03}%)'
        TOC += f'    - [`{title:_<8}{name}`](#{id})\n'
        document+=f'Max Strength of Output: {strength:0.08}%\n\n'
        document+=f'Prompt Ids: {id}\n\n'
        for image in images:
            if not image.endswith(".nsfw"):
                document+=f'![{image}]({image})\n'
            else:
                document+=f'![{image}](nsfw.png)\n'

    # TOC += "- [Top 20 Shared Overfit Images(Strength | Difference)](#sharedoverfit)\n"
    # document += f'\n\n## Shared Overfit Images## {{: #sharedoverfit }}\n'
    # document += '\n\nThese are similar to the previous 20 in terms of the strength, however they label appears in other models outputs as well. If the two images are not similar at all, then the image should be treated the same as if it was in the group above. If the images are similar than the two models are likely family (connected through a chain of models retrained off of eachoter). The model similarity table can provide hints to this. The higher the similarity count and lower the average similarity the close the two models are to eachother in this chain. Whichever model was created first, would be the model that the training image exists in.\n\nWhy care: The other model may be easier to get a copy of, or it\'s data may be easier to access. Allowing for a proxy attack.\n\n'
    # for pairs in shared[:20]:
    #     name,strength,_ = pairs[0]
    #     pairs = pairs[1:]
    #     pairs.sort(key=lambda d:d[0])
    #     document+=f'### {name}### {{: #{name} }}\n'
    #     title = f'({strength:0.03}% | {pairs[0][0]*100:0.05}%)'
    #     TOC += f'    - [`{title:_<18}{name}`](#{name})\n'
    #     document+=f'Max Strength of Output: {strength:0.08}%\n\n'
    #     for pair in pairs:
    #         document+=f'Difference of **{pair[0]*100:0.08}%** compared to **{pair[3]}**\n\n'
    #         if not pair[1].endswith(".nsfw"):
    #             document+=f'![{pair[1]}]({pair[1]})'
    #         else:
    #             document+=f'![{pair[1]}](nsfw.png)'
    #         if not pair[2].endswith(".nsfw"):
    #             document+=f' ![{pair[2]}]({pair[2]}) \n\n'
    #         else:
    #             document+=f' ![{pair[2]}](nsfw.png) \n\n'

    # TOC += "- [Top 20 Most Similar Images(Strength | Difference)](#mostsimilar)\n"
    # document += f'\n\n## Shared Overfit Images## {{: #mostsimilar }}\n'
    # document += '\n\nThese are the most similar images. This can be useful when destingquising which model is the child and which is the parrent/grandparent.\n\n'
    # shared.sort(key=lambda d:d[0][2])
    # for pairs in shared[:20]:
    #     name,strength,_ = pairs[0]
    #     pairs = pairs[1:]
    #     pairs.sort(key=lambda d:d[0])
    #     document+=f'### {name}### {{: #{name} }}\n'
    #     title = f'({strength:0.03}% | {pairs[0][0]*100:0.05}%)'
    #     TOC += f'    - [`{title:_<18}{name}`](#{name})\n'
    #     document+=f'Max Strength of Output: {strength:0.08}%\n\n'
    #     for pair in pairs:
    #         document+=f'Difference of **{pair[0]*100:0.08}%** compared to **{pair[3]}**\n\n'
    #         if not pair[1].endswith(".nsfw"):
    #             document+=f'![{pair[1]}]({pair[1]})'
    #         else:
    #             document+=f'![{pair[1]}](nsfw.png)'
    #         if not pair[2].endswith(".nsfw"):
    #             document+=f' ![{pair[2]}]({pair[2]}) \n\n'
    #         else:
    #             document+=f' ![{pair[2]}](nsfw.png) \n\n'
    f = open(model+'.html','w')
    document = TOC+'\n\n'+document
    f.write(markdown.markdown(document,extensions=['attr_list','tables']))
    f.close()

exit()

handMeDowns = {}
novels = {}
for newF in newFs:
    if newF.endswith('.py'):
        continue
    name = newF.split("-")[1]
    if name == '':
        name="????????????"
    ids = newF.split("-")[2]
    if ids in originalVocabs.keys():
        for originalVocabF in originalVocabs[ids]:
            score = compare(
                        os.path.join(new,newF), 
                        os.path.join(original,originalVocabF)
                    )
            data = [
                    os.path.join(original,originalVocabF),
                    os.path.join(new,newF), 
                    score,
                    float(originalVocabF.split("-")[0])*percentMultiplier,
                    float(newF.split("-")[0])*percentMultiplier,
                    name,
                    ids
                ]
            try:
                handMeDowns[ids].append(data)
                handMeDowns[ids][0] = min(handMeDowns[ids][0],score)
            except KeyError:
                handMeDowns[ids] = [score,data]
    else:
        score = float(newF.split("-")[0])
        data = [
            os.path.join(new,newF),
            score*percentMultiplier,
            name,
            ids
        ]
        try:
            novels[ids].append(data)
            novels[ids][0] = max(novels[ids][0],score)
        except KeyError as e:
            novels[ids] = [score,data]




document=""

TOC = ""
if original is not None:
    TOC += "## Table of Contents\n- [Inheritted Overfit Images (difference)](#inherittedoverfit)\n"
    document += f'## Inheritted Overfit Images## {{: #inherittedoverfit }}, average difference ignores this truncation\n'

handMeDowns = list(handMeDowns.values())
handMeDowns.sort(key=lambda d: d[0])
numSimilar = 0
differenceSum = 0
block = ""
for pairs in handMeDowns:
    pairs = pairs[1:]
    pairs.sort(key=lambda d:d[2])
    block+=f'### {pairs[0][5]}### {{: #{pairs[0][6]} }}\n'
    if pairs[0][2]*100 < 5:
        TOC += f'    - [{pairs[0][5]} ({pairs[0][2]*100:0.03}%)](#{pairs[0][6]})\n'
    done = []
    for handMeDown in pairs:
        if handMeDown[0] in done or handMeDown[1] in done:
            continue
        change = str(handMeDown[4]-handMeDown[3])
        if change[0] != '-':
            change = '+'+change
        block+=f'###### Strength of Output: {handMeDown[4]:0.08}% ({change:0.08}%)\n'
        block+=f'###### Difference: {handMeDown[2]*100:0.08}%\n'
        if not handMeDown[0].endswith(".nsfw"):
            block+=f'![{handMeDown[0]}]({handMeDown[0]})'
        else:
            block+=f'![{handMeDown[0]}](nsfw.png)'
        if not handMeDown[1].endswith(".nsfw"):
            block+=f' ![{handMeDown[1]}]({handMeDown[1]}) \n'
        else:
            block+=f' ![{handMeDown[1]}](nsfw.png) \n'
        done.append(handMeDown[0])
        done.append(handMeDown[1])
        numSimilar +=1
        differenceSum += handMeDown[2]

if len(handMeDowns) > 0:
    TOC = f'#### Average difference: {differenceSum/numSimilar*100:0.08}\n'+TOC
    TOC = f'#### Inheritted Overfit Images: {numSimilar}\n' + TOC
document += block+'\n'

TOC += "- [ Novel Overfit Images (output strength)]( #noveloverfit )\n"
document += f'## Novel Overfit Images## {{: #noveloverfit }}\n'
novels = list(novels.values())
novels.sort(key=lambda d:d[0], reverse=True)
for l in novels:
    l = l[1:]
    l.sort(reverse=True, key=lambda d:d[1])
    document += f'### {l[0][2]}### {{: #{l[0][3]} }}\n'
    if l[0][1] > 10:
        TOC += f'    - [{l[0][2]} ({l[0][1]:0.08}%)](#{l[0][3]})\n'
    for novel in l:
        document += f'###### Strength of Output: {novel[1]}%\n'
        if not novel[0].endswith(".nsfw"):
            document += f'![{novel[0]}]({novel[0]})\n'
        else:
            document += f'![{novel[0]}](nsfw.png)\n'

if original is not None:
    TOC += '\n\n*Inherited TOC Truncated at 5%, unlisted matches still can be viewed bellow though*\n'
TOC += '\n\n*Novel TOC Truncated at 10%, strength down to 5% viewable bellow*\n'
document = TOC+document
if original is None:
    document = f"# {new}\n"+document
else:
    document = f"# {original} -> {new}\n"+document
f = open(f'{new} from {original}.html','w')
f.write(markdown.markdown(document,extensions=['attr_list']))
f.close()