import os
import time
import argparse

import torch
import torch._inductor
from torch.export import export

torch.set_float32_matmul_precision('high')


def _generate_test_data(n_events) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate dummy event data for testing."""
    ctx = torch.rand(n_events, 4).cuda() * 0.9
    ctx[:, 3] = torch.randint(0, 2, (n_events,)).cuda()
    cnt = torch.tensor([n_events]).cuda()
    return ctx, cnt


def count_parameters(model):
    """Count total parameters in a model"""
    return sum(p.numel() for p in model.parameters())


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Export F3 model to PT2 format with configurable options')
    
    parser.add_argument('-m', '--model-name', type=str, 
                       default='1280x720x20_patchff_ds1_small',
                       help='Model name/configuration to load (default: 1280x720x20_patchff_ds1_small)')
    
    parser.add_argument('--build-hashed-feats', action='store_true',
                       help='Build hashed features (default: True)')
    
    parser.add_argument('-n', '--n-events', type=int, default=200000,
                       help='Number of events to generate for static inference/profiling (default: 200000)')
    
    parser.add_argument('-o', '--output-name', type=str, default='f3_aoti.pt2',
                       help='Output PT2 file name (default: f3_aoti.pt2)')
    
    parser.add_argument('--warmup-runs', type=int, default=20,
                       help='Number of warmup runs (default: 20)')
    
    parser.add_argument('--runs', type=int, default=200,
                       help='Number of timed inference runs (default: 200)')
    
    parser.add_argument('-v', '--variable-events', action='store_true',
                       help='Compile models to be compatible with variable event counts')
    
    parser.add_argument('-c', '--checkpoint', type=str, default=None,
                       help='Path to checkpoint file (.pth) to load model weights from')

    return parser.parse_args()


def compile_engine_and_infer(args):
    model = torch.hub.load('grasp-lyrl/fast-feature-fields',
                           'f3',
                           name=args.model_name,
                           compile=False,
                           pretrained=False,
                           return_feat=True,
                           return_logits=False,
                           single_batch_mode=True).cuda().eval()

    # This is a hack to load compiled weights into a non-compiled model
    # This is an artifact of the fact, that we do not save the non-compiled model
    model_compiled = torch.compile(
        model,
        fullgraph=True,
        backend='inductor'
    )

    if args.checkpoint is not None:
        model_compiled.load_state_dict(
            torch.load(args.checkpoint, map_location="cpu", weights_only=True)['model']
        )
        print(f"Loaded F3 checkpoint from {args.checkpoint}")

    if args.build_hashed_feats:
        model._build_hashed_feats()
        print("Built hashed feats")
    else:
        print("Skipped building hashed feats")

    print(f"Model has {count_parameters(model):,} parameters")
    print(f"MultiHashEncoder has {count_parameters(model.multi_hash_encoder):,} parameters")
    print(f"Rest of the NN has {count_parameters(model) - count_parameters(model.multi_hash_encoder):,} parameters")

    ctx, _ = _generate_test_data(args.n_events)

    dynamic_shapes = None
    if args.variable_events:
        event_count = torch.export.Dim("event_count")
        dynamic_shapes = {"currentBlock": {0: event_count}}
    ep_model = export(model, (ctx,), dynamic_shapes=dynamic_shapes)

    with torch.autocast(enabled=True, device_type='cuda', dtype=torch.float16):
        with torch.no_grad():
            pt2_path = torch._inductor.aoti_compile_and_package(
                ep_model,
                package_path=os.path.join(os.getcwd(), args.output_name),
                inductor_configs={
                    "epilogue_fusion": True,
                    "max_autotune": True,
                }
            )

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
    print(f"     Model name: {args.model_name}")
    print(f"     Build hashed feats: {args.build_hashed_feats}")
    print(f"     Output name: {args.output_name}")
    print(f"     Warmup runs: {args.warmup_runs}")
    print(f"     Timed runs: {args.runs}")
    print(f"     Inference results for: {args.n_events:,} events")
    print(f"     Compile for variable events: {args.variable_events}")
    print(f"     Checkpoint file: {args.checkpoint}")
    print()
    
    compile_engine_and_infer(args)
