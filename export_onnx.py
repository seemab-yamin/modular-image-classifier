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
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        dynamo=False,
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
                return {"input": next(self.data_iter).numpy()}
            except StopIteration:
                return None

    # iterate over the validation dataset and collect samples for calibration
    calibration_samples = []
    for i, (inputs, labels) in enumerate(val_loader):
        calibration_samples.append(inputs)
        if i >= 10:  # Limit to 10 batches for calibration
            break

    calibration_reader = CalibrationDataReader(calibration_samples)
    # Apply static post-training quantization
    quantize_static(
        model_input=onnx_fp32_path,
        model_output=onnx_int8_static_path,
        calibration_data_reader=calibration_reader,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
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
        resize_size=(224, 224) if args.arch == "vit" else None,
    )

    # export the model to ONNX format
    export_model_fp8_dynamic_to_onnx(
        onnx_fp32_path=args.onnx_fp32_path,
        onnx_int8_dynamic_path=args.onnx_int8_dynamic_path,
    )

    export_model_fp8_static_to_onnx(
        onnx_fp32_path=args.onnx_fp32_path,
        onnx_int8_static_path=args.onnx_int8_static_path,
        val_loader=val_loader,
    )

    # load onnx models
    import onnxruntime as ort
    import torch

    from data_loader import make_dataloaders
    from eval import evaluate, save_confusion_matrix, save_summary_report

    providers = ["CPUExecutionProvider"]
    if torch.cuda.is_available():
        providers.append("CUDAExecutionProvider")

    onnx_fp32_path = "/content/drive/MyDrive/ai-projects/modular-image-classifier/artifacts/part_2_results/vit_cifar10.onnx"
    train_loader, val_loader, info = make_dataloaders(
        dataset_name="cifar10",
        batch_size=1,
        data_dir="/content/drive/MyDrive/ai-projects/modular-image-classifier/data",
        resize_size=(224, 224),
    )
    session_fp32 = ort.InferenceSession(onnx_fp32_path, providers=providers)
    input_name = session_fp32.get_inputs()[0].name
    output_name = session_fp32.get_outputs()[0].name

    import time

    val_start = time.time()
    all_preds = []
    all_labels = []
    for images, labels in val_loader:
        images = images.cpu().numpy()  # Convert to numpy array for ONNX Runtime
        outputs = session_fp32.run([output_name], {input_name: images})

        output_tensor = torch.from_numpy(
            outputs[0]
        )  # Convert back to tensor for evaluation
        _, predicted = output_tensor.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    val_time = time.time() - val_start
    val_acc = (
        100
        * sum(1 for pred, label in zip(all_preds, all_labels) if pred == label)
        / len(all_labels)
    )

    metrics = evaluate(all_labels, all_preds, info.class_names)

    import os

    artifacts_dir = (
        "/content/drive/MyDrive/ai-projects/modular-image-classifier/artifacts"
    )
    arch = "vit"
    dataset = "cifar10"

    cm_path = os.path.join(artifacts_dir, "part_2_results", f"cm_{arch}_{dataset}.png")
    save_confusion_matrix(
        metrics["confusion_matrix"],
        info.class_names,
        cm_path,
        title=f"Confusion Matrix - {arch} Model on {dataset}",
    )

    summary_path = os.path.join(artifacts_dir, "part_2_results", "summary.txt")
    save_summary_report(
        metrics["report"],
        info.class_names,
        arch,
        dataset,
        summary_path,
    )
