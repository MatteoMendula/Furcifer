import time 

def mobilenetv3_split_5_head_inference(image_unqueezed, model):
    result = {}
    start = time.time()
    model.eval()
    output_head = model(image_unqueezed)
    end = time.time()

    result['head_inference_result'] = output_head
    result['head_inference_time'] = end-start

    return result