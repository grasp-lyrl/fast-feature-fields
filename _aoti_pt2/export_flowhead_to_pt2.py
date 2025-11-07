import os
import time
import argparse

import torch
import torch._inductor
from torch.export import export

from f3.tasks.optical_flow.utils.models.model import FlowHead

torch.set_float32_matmul_precision('high')


def _generate_test_data(input_channels: int, height: int, width: int):
    """Generate random test data for inference"""
    ctx = torch.rand(1, input_channels, height, width).cuda().float().contiguous()
    return ctx


def count_parameters(model):
    """Count total parameters in a model"""
    return sum(p.numel() for p in model.parameters())


def load_checkpoint_weights(model, checkpoint_path):
    """Load checkpoint weights into the FlowHead model

    Args:
        model: FlowHead model instance
        checkpoint_path: Path to the checkpoint file
    """
    model_compiled = torch.compile(model, fullgraph=True)
    saved_state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)['model']
    new_state_dict = {}
    for k, v in saved_state_dict.items():
        if k.startswith('flowhead.'):
            new_k = k[len('flowhead.'):]
            new_state_dict[new_k] = v
    model_compiled.load_state_dict(new_state_dict)
    print(f"Loaded FlowHead ckpt from {checkpoint_path}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Export FlowHead model to PT2 format with configurable options')
    
    parser.add_argument('-i', '--input-channels', type=int, default=32,
                       help='Number of input channels (default: 32)')
    
    parser.add_argument('--height', type=int, default=238,
                       help='Input height (default: 238)')
    
    parser.add_argument('--width', type=int, default=308,
                       help='Input width (default: 308)')
    
    parser.add_argument('--kernels', type=int, nargs='+', default=[9, 9, 9, 9],
                       help='Kernel sizes for decoder blocks (default: [9, 9, 9, 9])')
    
    parser.add_argument('--btlncks', type=int, nargs='+', default=[2, 2, 2, 2],
                       help='Bottleneck factors for decoder blocks (default: [2, 2, 2, 2])')
    
    parser.add_argument('--dilations', type=int, nargs='+', default=[1, 1, 1, 1],
                       help='Dilation rates for decoder blocks (default: [1, 1, 1, 1])')
    
    parser.add_argument('-c', '--checkpoint', type=str, default=None,
                       help='Path to checkpoint file (.pth) to load model weights from')
    
    parser.add_argument('-o', '--output-name', type=str, default='flowhead_aoti.pt2',
                       help='Output PT2 file name (default: flowhead_aoti.pt2)')
    
    parser.add_argument('--warmup-runs', type=int, default=20,
                       help='Number of warmup runs (default: 20)')
    
    parser.add_argument('--runs', type=int, default=200,
                       help='Number of timed inference runs (default: 200)')
    
    parser.add_argument('--autocast', type=str, default=None,
                       choices=['float16', 'bfloat16', None],
                       help='Autocast dtype for compilation (float16, bfloat16, or None for no autocast)')

    return parser.parse_args()


def compile_engine_and_infer(args):
    flowhead_config = {
        'kernels': args.kernels,
        'btlncks': args.btlncks,
        'dilations': args.dilations
    }

    model = FlowHead(
        input_channels=args.input_channels,
        config=flowhead_config
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

    print("⚙️  Configuration:")
    print(f"     Input channels: {args.input_channels}")
    print(f"     Input shape: 1x{args.input_channels}x{args.height}x{args.width}")
    print(f"     Decoder kernels: {args.kernels}")
    print(f"     Decoder bottlenecks: {args.btlncks}")
    print(f"     Decoder dilations: {args.dilations}")
    print(f"     Checkpoint: {args.checkpoint}")
    print(f"     Output name: {args.output_name}")
    print(f"     Warmup runs: {args.warmup_runs}")
    print(f"     Timed runs: {args.runs}")
    print(f"     Autocast: {args.autocast if args.autocast else 'None'}")
    print()

    compile_engine_and_infer(args)
