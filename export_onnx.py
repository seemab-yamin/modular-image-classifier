import torch


def export_model_fp32__to_onnx(
    model, onnx_path, batch_size, channels, height, width, device
):
    """
    Exports the trained PyTorch model to ONNX format fp32.
    """

    dummy_input = torch.randn(batch_size, channels, height, width).to(device)
    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=18,
        do_constant_folding=True,  # Reduces model size and speeds up inference
        verbose=False,  # Set to True for detailed export logs
    )


def export_model_fp32_cleaned_to_onnx(onnx_path, cleaned_onnx_path):
    """
    Cleans the exported ONNX model by removing unnecessary value_info entries.
    """
    import onnx

    # Load the ONNX model
    onnx_model = onnx.load(onnx_path)

    # Remove unnecessary value_info entries
    while len(onnx_model.graph.value_info) > 0:
        onnx_model.graph.value_info.pop()

    # Save the cleaned ONNX model
    onnx.save(onnx_model, cleaned_onnx_path)


def export_model_int8_dynamic_to_onnx(onnx_fp32_path, onnx_int8_dynamic_path):
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


def export_model_int8_static_to_onnx(onnx_fp32_path, onnx_int8_static_path, val_loader):
    """
    Exports the trained PyTorch model to ONNX format fp8.
    """
    from onnxruntime.quantization import (
        CalibrationDataReader,
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


def evaluate_onnx_model(onnx_path, val_loader, artifacts_dir):
    import onnxruntime as ort
    import torch

    from eval import evaluate, save_confusion_matrix, save_summary_report

    providers = ["CPUExecutionProvider"]
    if torch.cuda.is_available():
        providers.append("CUDAExecutionProvider")

    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    all_preds = []
    all_labels = []
    for images, labels in val_loader:
        images = images.cpu().numpy()  # Convert to numpy array for ONNX Runtime
        outputs = session.run([output_name], {input_name: images})

        output_tensor = torch.from_numpy(
            outputs[0]
        )  # Convert back to tensor for evaluation
        _, predicted = output_tensor.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    val_acc = (
        100
        * sum(1 for pred, label in zip(all_preds, all_labels) if pred == label)
        / len(all_labels)
    )
    metrics = evaluate(all_labels, all_preds, info.class_names)

    cm_path = os.path.join(
        artifacts_dir, "part_2_results", f"{onnx_path.split('/')[-1].split('.')[0]}.png"
    )
    save_confusion_matrix(
        metrics["confusion_matrix"],
        info.class_names,
        cm_path,
        title=f"Confusion Matrix - {onnx_path.split('/')[-1].split('.')[0]}",
    )

    summary_path = os.path.join(artifacts_dir, "part_2_results", "summary.txt")
    save_summary_report(
        metrics["report"],
        info.class_names,
        "",
        "",
        summary_path,
    )

    return val_acc, metrics, cm_path, summary_path


def export_model_to_onnx(onnx_fp32_path, artifacts_dir, arch, dataset, val_loader):
    cleaned_onnx_fp32_path = os.path.join(
        artifacts_dir,
        "part_2_results",
        f"cleaned_{arch}_fp32_{dataset}.onnx",
    )
    onnx_int8_dynamic_path = os.path.join(
        artifacts_dir,
        "part_2_results",
        f"{arch}_int8_dynamic_{dataset}.onnx",
    )
    onnx_int8_static_path = os.path.join(
        artifacts_dir,
        "part_2_results",
        f"{arch}_int8_static_{dataset}.onnx",
    )
    export_model_fp32_cleaned_to_onnx(onnx_fp32_path, cleaned_onnx_fp32_path)
    print(f"✅ Cleaned ONNX model saved to: {cleaned_onnx_fp32_path}")
    export_model_int8_dynamic_to_onnx(cleaned_onnx_fp32_path, onnx_int8_dynamic_path)
    print(f"✅ Exported INT8 dynamic model to ONNX: {onnx_int8_dynamic_path}")
    export_model_int8_static_to_onnx(onnx_fp32_path, onnx_int8_static_path, val_loader)
    print(f"✅ Exported INT8 static model to ONNX: {onnx_int8_static_path}")


if __name__ == "__main__":
    # read a validation dataset
    import argparse
    import os

    from data_loader import make_dataloaders

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Name of the dataset")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--augment", action="store_true", help="Enable augmentation")
    parser.add_argument("--arch", type=str, help="Model architecture")
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="./artifacts",
        help="Directory to save artifacts",
    )

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
    parser.add_argument(
        "--export-model-to-onnx",
        action="store_true",
        default=False,
        help="Export the model to ONNX format",
    )
    # evaluate onnx models for all list
    parser.add_argument(
        "--evaluate-onnx-models",
        action="store_true",
        default=False,
        help="Evaluate ONNX models",
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
    if args.export_model_to_onnx:
        export_model_to_onnx(
            onnx_fp32_path=args.onnx_fp32_path,
            artifacts_dir=args.artifacts_dir,
            arch=args.arch,
            dataset=args.dataset,
            val_loader=val_loader,
        )

    if args.evaluate_onnx_models:
        for onnx_model in args.evaluate_onnx_models:
            evaluate_onnx_model(
                onnx_path=onnx_model,
                val_loader=val_loader,
                artifacts_dir=args.artifacts_dir,
            )
