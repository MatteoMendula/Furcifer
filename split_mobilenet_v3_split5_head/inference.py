def mobilenetv3_split_5_head_inference(image_unqueezed, model):
    output_head = model(image_unqueezed)

    return output_head