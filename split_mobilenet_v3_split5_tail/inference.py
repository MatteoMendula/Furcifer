def mobilenetv3_split_5_tail_inference(head_inference_result, model):
    output_head = model(head_inference_result)

    return output_head