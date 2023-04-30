import torch
import time

def mobilenetv3_split_5_tail_inference(head_inference_result, model):
    result = {}

    start = time.time()
    model.eval()
    output_head = model(head_inference_result)

    classes_probabilies = {}

    for idx in torch.topk(output_head, k=3).indices.squeeze(0).tolist():
        prob = torch.softmax(output_head, dim=1)[0, idx].item()
        classes_probabilies[idx] = prob*100

    end = time.time()

    result['tail_inference_result'] = classes_probabilies
    result['tail_inference_time'] = end - start

    return result