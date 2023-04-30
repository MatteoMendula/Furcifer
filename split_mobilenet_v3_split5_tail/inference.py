import torch

def mobilenetv3_split_5_tail_inference(head_inference_result, model):
    output_head = model(head_inference_result)

    classes_probabilies = {}

    for idx in torch.topk(output_head, k=3).indices.squeeze(0).tolist():
        prob = torch.softmax(output_head, dim=1)[0, idx].item()
        classes_probabilies[idx] = prob*100


    return classes_probabilies