import torch
import pickle
from PIL import Image
import torchvision.transforms as transforms


def get_coco_object_dictionary():
    import os
    file_with_coco_names = "category_names.txt"

    if not os.path.exists(file_with_coco_names):
        print("Downloading COCO annotations.")
        import urllib
        import zipfile
        import json
        import shutil
        urllib.request.urlretrieve("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", "cocoanno.zip")
        with zipfile.ZipFile("cocoanno.zip", "r") as f:
            f.extractall()
        print("Downloading finished.")
        with open("annotations/instances_val2017.json", 'r') as COCO:
            js = json.loads(COCO.read())
        class_names = [category['name'] for category in js['categories']]
        open("category_names.txt", 'w').writelines([c+"\n" for c in class_names])
        os.remove("cocoanno.zip")
        shutil.rmtree("annotations")
    else:
        class_names = open("category_names.txt").readlines()
        class_names = [c.strip() for c in class_names]
    return class_names

def plot_results(best_results, inputs, classes_to_labels):
    from matplotlib import pyplot as plt
    import matplotlib.patches as patches
    fig, ax = plt.subplots(1)
    # Show original, denormalized image...
    print(inputs.shape)
    inputs = torch.transpose(inputs.squeeze(0), 0, 2).transpose(0, 1)
    print(inputs.shape)
    ax.imshow(inputs)
    # ...with detections
    bboxes = best_results[1][0]["boxes"].cpu().detach().numpy().tolist()
    classes = best_results[1][0]["labels"].cpu().detach().numpy().tolist()
    confidences = best_results[1][0]["scores"].cpu().detach().numpy().tolist()
    for idx in range(len(bboxes)):
        if confidences[idx] < 0.7:
            continue
        left, bot, right, top = bboxes[idx]
        x, y, w, h = [val for val in [left, bot, right - left, top - bot]]
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y, "{} {:.0f}%".format(classes_to_labels[classes[idx] - 1], confidences[idx]*100), bbox=dict(facecolor='white', alpha=0.5))
    plt.show()

head_model_loaded = torch.jit.load('./models/head_model.pt')
tail_model_loaded = torch.jit.load('./models/tail_model.pt')

image = Image.open("kitti_1.png")
transform = transforms.Compose([
    transforms.ToTensor()
])

image = transform(image)
my_image = torch.unsqueeze(torch.Tensor(image), 0)
input = torch.tensor(my_image, dtype=torch.float32)
input = input.cuda()

out_head, other_info = head_model_loaded(input)
print("----------- head --------------")
print(out_head)
print(other_info)
print("----------- tail --------------")
out_tail_loaded = tail_model_loaded(*(*out_head, *other_info))

print(out_tail_loaded)

classes_to_labels= get_coco_object_dictionary()   
plot_results(out_tail_loaded, input.cpu(), classes_to_labels)