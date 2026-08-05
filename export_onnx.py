import torch


def export_model_fp32__to_onnx(
    model, onnx_path, batch_size, channels, height, width, device
):
    """
    Exports the trained PyTorch model to ONNX format fp32.
    """

    dummy_input = torch.randn(batch_size, channels, height, width).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_shapes={"input": {0: torch.export.Dim("batch_size")}},
        opset_version=18,
        do_constant_folding=True,  # Reduces model size and speeds up inference
        verbose=False,  # Set to True for detailed export logs
    )


def export_model_fp8_dynamic_to_onnx(onnx_fp32_path, onnx_int8_dynamic_path):
    """
    Exports the trained PyTorch model to ONNX format fp8.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    # Apply dynamic post-training quantization
    quantize_dynamic(
        model_input=onnx_fp32_path,
        model_output=onnx_int8_dynamic_path,
        weight_type=QuantType.QUInt8,  # Quantize weights to unsigned INT8
    )


def export_model_fp8_static_to_onnx(onnx_fp32_path, onnx_int8_static_path, val_loader):
    """
    Exports the trained PyTorch model to ONNX format fp8.
    """
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    class CalibrationDataReader(CalibrationDataReader):
        def __init__(self, calibration_data):
            self.data = calibration_data
            self.data_iter = iter(self.data)

        def get_next(self):
            try:
                return next(self.data_iter)
            except StopIteration:
                return None

    # iterate over the validation dataset and collect samples for calibration
    calibration_samples = []
    for i, (inputs, labels) in enumerate(val_loader):
        calibration_samples.append(inputs)
        if i >= 10:  # Limit to 10 batches for calibration
            break

    # Apply static post-training quantization
    quantize_static(
        model_input=onnx_fp32_path,
        model_output=onnx_int8_static_path,
        weight_type=QuantType.QUInt8,  # Quantize weights to unsigned INT8
    )


if __name__ == "__main__":
    # read a validation dataset
    import argparse

    from data_loader import make_dataloaders

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Name of the dataset")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--augment", action="store_true", help="Enable augmentation")
    parser.add_argument(
        "--data-dir", type=str, default="./data", help="Data directory for the dataset"
    )
    parser.add_argument(
        "--onnx-fp32-path",
        type=str,
        default="model_fp32.onnx",
        help="Path to save the FP32 ONNX model",
    )
    parser.add_argument(
        "--onnx-int8-dynamic-path",
        type=str,
        default="model_fp8_dynamic.onnx",
        help="Path to save the FP8 dynamic ONNX model",
    )
    parser.add_argument(
        "--onnx-int8-static-path",
        type=str,
        default="model_fp8_static.onnx",
        help="Path to save the FP8 static ONNX model",
    )
    args = parser.parse_args()

    train_loader, val_loader, info = make_dataloaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=args.augment,
        data_dir=args.data_dir,
    )
    export_model_fp8_dynamic_to_onnx(
        onnx_fp32_path=args.onnx_fp32_path,
        onnx_int8_dynamic_path=args.onnx_int8_dynamic_path,
    )

    export_model_fp8_static_to_onnx(
        onnx_fp32_path=args.onnx_fp32_path,
        onnx_int8_static_path=args.onnx_int8_static_path,
        val_loader=val_loader,
    )
