import json
import torch
from torchvision import transforms
from PIL import Image
import time
import glob

from efficientnet_pytorch import EfficientNet

def efficient_net_inference(img_pil, model, processor = 'gpu'):


    delay_loading = 0
    if type(model) == str:
        if processor == 'gpu':
            start=time.time()
            model = EfficientNet.from_pretrained(model)
            model.cuda()
            torch.cuda.synchronize()
            end= time.time()
        else:
            start=time.time()
            model = EfficientNet.from_pretrained(model)
            end= time.time()        

        delay_loading = (end-start)

   
    # Preprocess image
    tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

    img = tfms(img_pil).unsqueeze(0)
    #print(img.shape) # torch.Size([1, 3, 224, 224])

    # Load ImageNet class names
    labels_map = json.load(open('./labels_map.txt'))
    labels_map = [labels_map[str(i)] for i in range(1000)]

    # Classify
    if processor == 'gpu':
        print("using gpu")
        start=time.time()
        model.eval()
        with torch.no_grad():
            outputs = model(img.cuda())
        end=time.time()
    else:
        print("using cpu")
        start=time.time()
        model.eval()
        with torch.no_grad():
            outputs = model(img)
        end=time.time()

    delay_inference = (end-start)

    classes_probabilies = {}

    # PREDICTIONS
    for idx in torch.topk(outputs, k=3).indices.squeeze(0).tolist():
        prob = torch.softmax(outputs, dim=1)[0, idx].item()
        prob_string = '{label:<75} ({p:.2f}%)'.format(label=labels_map[idx], p=prob*100)
        classes_probabilies[labels_map[idx]] = prob*100
    
    del model

    result = dict()
    result["delay_loading"] = delay_loading
    result["delay_inference"] = delay_inference
    result["classes_probabilies"] = classes_probabilies

    return result
    
if __name__ == "__main__":
    img_pil = Image.open('000000001675.jpg')
    res=efficient_net_inference(img_pil, 'efficientnet-b0','gpu') # or efficientnet-b1

    for key in res:
        print(key,": ",res[key])
        
        

