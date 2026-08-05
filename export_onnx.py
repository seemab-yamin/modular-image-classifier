import torch


def export_model_to_onnx(model, onnx_path, batch_size, channels, height, width, device):
    """
    Exports the trained PyTorch model to ONNX format.
    """
    dummy_input = torch.randn(batch_size, channels, height, width).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=17,
        do_constant_folding=True,  # Reduces model size and speeds up inference
        verbose=False,  # Set to True for detailed export logs
    )
