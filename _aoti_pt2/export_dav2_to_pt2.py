import os
import time
import argparse

import torch
import torch._inductor
from torch.export import export
from f3.tasks.depth.utils.models import FFDepthAnythingV2

torch.set_float32_matmul_precision('high')


def _generate_test_data(input_channels: int, height: int, width: int):
    """Generate random test data for inference"""
    ctx = torch.rand(1, input_channels, height, width).cuda().float().contiguous()
    return ctx


def count_parameters(model):
    """Count total parameters in a model"""
    return sum(p.numel() for p in model.parameters())


def load_checkpoint_weights(model, checkpoint_path):
    """Load checkpoint weights into the model, handling the 'dav2.' prefix
    
    Args:
        model: FFDepthAnythingV2 model instance
        checkpoint_path: Path to the checkpoint file
    """
    saved_state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)['model']
    new_state_dict = {}
    for k, v in saved_state_dict.items():
        if k.startswith('dav2.'):
            new_k = k[len('dav2.'):]
            new_state_dict[new_k] = v
    model.dav2.load_state_dict(new_state_dict)
    print(f"Loaded DepthAnythingV2 ckpt from {checkpoint_path}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Export DepthAnythingV2 model to PT2 format with configurable options')
    
    parser.add_argument('-e', '--encoder', type=str, 
                       default='vitb',
                       choices=['vits', 'vitb', 'vitl', 'vitg'],
                       help='Encoder architecture (default: vitb)')
    
    parser.add_argument('-i', '--input-channels', type=int, default=32,
                       help='Number of input channels (default: 32)')
    
    parser.add_argument('--height', type=int, default=238,
                       help='Input height (default: 238)')
    
    parser.add_argument('--width', type=int, default=308,
                       help='Input width (default: 308)')
    
    parser.add_argument('-c', '--checkpoint', type=str, default=None,
                       help='Path to checkpoint file (.pth) to load model weights from')
    
    parser.add_argument('-o', '--output-name', type=str, default='dav2_aoti.pt2',
                       help='Output PT2 file name (default: dav2_aoti.pt2)')
    
    parser.add_argument('--warmup-runs', type=int, default=20,
                       help='Number of warmup runs (default: 20)')
    
    parser.add_argument('--runs', type=int, default=200,
                       help='Number of timed inference runs (default: 200)')
    
    parser.add_argument('--autocast', type=str, default=None,
                       choices=['float16', 'bfloat16', None],
                       help='Autocast dtype for compilation (float16, bfloat16, or None for no autocast)')

    return parser.parse_args()


def compile_engine_and_infer(args):
    model = FFDepthAnythingV2(
        dav2_config={
            'encoder': args.encoder,
        },
        input_channels=args.input_channels
    ).cuda().eval()

    if args.checkpoint is not None:
        load_checkpoint_weights(model, args.checkpoint)

    print(f"Model has {count_parameters(model):,} parameters")

    ctx = _generate_test_data(args.input_channels, args.height, args.width)

    ep_model = export(model, (ctx,))

    # Determine autocast settings
    autocast_dtype = None
    if args.autocast == 'float16':
        autocast_dtype = torch.float16
    elif args.autocast == 'bfloat16':
        autocast_dtype = torch.bfloat16

    with torch.autocast(enabled=autocast_dtype is not None, device_type='cuda', dtype=autocast_dtype):
        with torch.no_grad():
            pt2_path = torch._inductor.aoti_compile_and_package(
                ep_model,
                package_path=os.path.join(os.getcwd(), args.output_name),
                inductor_configs={
                    "epilogue_fusion": True,
                    "max_autotune": True,
                })

    # Get the size of the exported model
    model_size_bytes = os.path.getsize(pt2_path)
    model_size_mb = model_size_bytes / (1024 * 1024)
    print(f"Exported and compiled model to {pt2_path}")
    print(f"Model size: {model_size_mb:.2f} MB")

    ############################################################
    #                Inference benchmarking                    #
    ############################################################
    aoti_compiled = torch._inductor.aoti_load_package(pt2_path)

    # Warm-up runs
    for _ in range(args.warmup_runs):
        out = aoti_compiled(ctx)
    print(f"Output shape: {out.shape}")

    # Timed runs
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(args.runs):
        _ = aoti_compiled(ctx)
    torch.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    print(f"Average inference time over {args.runs} runs: {total_time / args.runs * 1000:.2f} ms")

    torch._dynamo.reset()


if __name__ == '__main__':
    args = parse_args()

    if args.autocast == "float16":
        print("⚠️ WARNING: Using float16 autocast with our pretrained DepthAnythingV2 models")
        print("   often does not work well and can lead to suboptimal results.")
        print("   Consider using bfloat16 or no autocast for good performance.")
        print()

    print("⚙️  Configuration:")
    print(f"     Encoder: {args.encoder}")
    print(f"     Input channels: {args.input_channels}")
    print(f"     Input shape: 1x{args.input_channels}x{args.height}x{args.width}")
    print(f"     Checkpoint: {args.checkpoint}")
    print(f"     Output name: {args.output_name}")
    print(f"     Warmup runs: {args.warmup_runs}")
    print(f"     Timed runs: {args.runs}")
    print(f"     Autocast: {args.autocast if args.autocast else 'None'}")
    print()

    compile_engine_and_infer(args)
