# export_single_onnx.py
import os
import argparse
import torch
import onnx

from models.generator import ConditionalSpriteGenerator
from utils.metadata_config import MetadataEncoder
from utils.util import load_config


def build_model(config_path: str, weights_path: str, device: str):
    config = load_config(config_path)
    encoder = MetadataEncoder(config)
    z_dim = config["train"]["z_dim"]

    model = ConditionalSpriteGenerator(z_dim, encoder.meta_dim).to(device)
    state_dict = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model, z_dim, encoder.meta_dim


def make_dummy_inputs(batch: int, z_dim: int, meta_dim: int, device: str):
    z = torch.randn(batch, z_dim, device=device, dtype=torch.float32)
    meta = torch.randn(batch, meta_dim, device=device, dtype=torch.float32)
    return z, meta


@torch.no_grad()
def sanity_forward(model, z, meta):
    out = model(z, meta)
    print("Sanity forward output shape:", tuple(out.shape))
    print("Output stats: mean=", out.mean().item(), "std=", out.std().item())
    return out


def export_single_file_onnx(
    model,
    inputs,
    out_path: str,
    opset: int,
    constant_folding: bool,
):
    """
    Forces a single ONNX file by disabling external data.
    """
    # Some torch versions accept this flag; some don't. We'll try it and fallback.
    try:
        torch.onnx.export(
            model,
            inputs,
            out_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=constant_folding,
            input_names=["z", "metadata"],
            output_names=["output"],
            dynamic_axes={
                "z": {0: "batch"},
                "metadata": {0: "batch"},
                "output": {0: "batch"},
            },
            # Force embedding weights into the .onnx instead of creating .onnx.data
            use_external_data_format=False,
            # Prefer legacy exporter to avoid opset bump + conversion dance
            dynamo=False,
            verbose=False,
        )
        return
    except TypeError:
        # Older torch: no use_external_data_format or no dynamo kwarg
        pass

    # Fallback export (older torch). This MAY still generate external data depending on size.
    torch.onnx.export(
        model,
        inputs,
        out_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=constant_folding,
        input_names=["z", "metadata"],
        output_names=["output"],
        dynamic_axes={
            "z": {0: "batch"},
            "metadata": {0: "batch"},
            "output": {0: "batch"},
        },
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
        verbose=False,
    )


def ensure_single_file(out_path: str):
    """
    If an external data file was created anyway, attempt to merge it back into one .onnx.
    This uses onnx.save_model with save_as_external_data=False.
    """
    data_path = out_path + ".data"
    if not os.path.exists(data_path):
        return

    print(f"\nDetected external data file: {data_path}")
    print("Attempting to merge external tensors into a single .onnx...")

    # Load with external data, then save with embedded tensors.
    model = onnx.load(out_path, load_external_data=True)

    # Save as a single file (no external data)
    # NOTE: this will produce a large .onnx file.
    onnx.save_model(
        model,
        out_path,
        save_as_external_data=False,
    )

    # If merge succeeded, we can delete the .data file
    # (Some environments may prefer to keep it; you can comment this out.)
    if os.path.exists(data_path):
        os.remove(data_path)
        print("Removed external data file after merge.")


def summarize_onnx(path: str):
    m = onnx.load(path, load_external_data=False)
    num_nodes = len(m.graph.node)
    num_inits = len(m.graph.initializer)
    init_has_external = any(len(t.external_data) > 0 for t in m.graph.initializer)

    init_bytes = sum(len(t.raw_data) for t in m.graph.initializer)
    # If tensors are stored in raw_data, init_bytes will be large.
    # If init_has_external is True, the weights are not embedded.

    print("\n=== ONNX summary ===")
    print("file:", path)
    print("size_bytes:", os.path.getsize(path))
    print("ir_version:", m.ir_version)
    print("opset_import:", [(o.domain, o.version) for o in m.opset_import])
    print("#nodes:", num_nodes)
    print("#initializers:", num_inits)
    print("initializer_raw_data_total_bytes:", init_bytes)
    print("has_external_data_refs:", init_has_external)
    print("inputs:", [i.name for i in m.graph.input])
    print("outputs:", [o.name for o in m.graph.output])

    if init_has_external:
        print(
            "\nWARNING: Model still references external data.\n"
            "You likely still have a .onnx.data file nearby, or your exporter saved external tensors."
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--weights", default="models/sprite_gen.pt")
    ap.add_argument("--out", default="models/sprite_generator_single.onnx")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--opset", type=int, default=14)
    ap.add_argument("--constant-folding", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    model, z_dim, meta_dim = build_model(args.config, args.weights, args.device)
    z, meta = make_dummy_inputs(args.batch, z_dim, meta_dim, args.device)
    sanity_forward(model, z, meta)

    export_single_file_onnx(
        model=model,
        inputs=(z, meta),
        out_path=args.out,
        opset=args.opset,
        constant_folding=args.constant_folding,
    )

    # If exporter still created .onnx.data, merge it into a single file.
    ensure_single_file(args.out)

    # Validate
    onnx.checker.check_model(onnx.load(args.out))
    summarize_onnx(args.out)
    print("\n✅ Done. Single-file ONNX written to:", args.out)


if __name__ == "__main__":
    main()
