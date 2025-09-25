import torch
from typing import Callable
from timeit import default_timer as timer

from f3 import init_event_model
from f3.utils import setup_torch

from f3.tasks.optical_flow.utils import EventFFFlow, EventFlow
from f3.tasks.depth.utils import EventFFDepthAnythingV2, EventDepthAnythingV2
from f3.tasks.segmentation.utils import EventFFSegformer, EventSegformer


class PerformanceTester:
    """Performance testing class for event-based models."""
    
    def __init__(self, config_path: str, width: int = 640, height: int = 480, 
                 timebins: int = 20, n_events: int = 200000, runs: int = 1000):
        """
        Initialize performance tester.
        
        Args:
            config_path: Path to model configuration file
            width: Image width
            height: Image height  
            timebins: Number of time bins
            n_events: Number of events to generate
            runs: Number of performance test runs
        """
        self.cfg = config_path
        self.width = width
        self.height = height
        self.timebins = timebins
        self.n_events = n_events
        self.runs = runs
        
        self.flowhead_config = {
            "kernels": [7, 7, 7, 7], 
            "btlncks": [2, 2, 2, 2], 
            "dilations": [1, 1, 1, 1]
        }
        
        self.depth_config = {
            "encoder": "vitb",
            "size": 518,
        }
        
        self.seg_config = "nvidia/segformer-b3-finetuned-cityscapes-1024-1024"
        
        setup_torch(cudnn_benchmark=True)
        self.cparams = torch.tensor([[0, 0, height, width]], dtype=torch.int32)
        self.ctx, self.cnt = self._generate_test_data()
        
    def _generate_test_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic event data for testing."""
        ctx = torch.rand(self.n_events, 4).cuda() * 0.9
        ctx[:, 3] = torch.randint(0, 2, (self.n_events,)).cuda()
        cnt = torch.tensor([self.n_events]).cuda()
        return ctx, cnt
    
    def _benchmark_model(self, test_func: Callable, warmup_runs: int = 20) -> float:
        """
        Benchmark a model's performance.
        
        Args:
            model: The model to benchmark
            test_func: Function that performs the forward pass
            warmup_runs: Number of warmup runs
            
        Returns:
            Average inference time in milliseconds
        """
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            with torch.no_grad():
                # Warmup runs
                for _ in range(warmup_runs):
                    test_func()
                
                # Actual benchmark
                torch.cuda.synchronize()
                start = timer()
                for _ in range(self.runs):
                    test_func()
                torch.cuda.synchronize()
                end = timer()
                
        return (end - start) / self.runs * 1000
    
    def test_eventff_model(self) -> dict[str, float]:
        """Test EventFF model performance."""
        print("="*60)
        print("F3 (EventFF) PERFORMANCE TESTING")
        print("="*60)
        
        results = {}
        
        model = init_event_model(self.cfg, return_feat=True).cuda()
        model = torch.compile(
            model,
            fullgraph=False,
            backend="inductor",
            options={
                "epilogue_fusion": True,
                "max_autotune": True,
            },
        )
        
        def feature_test():
            _, feat = model(self.ctx, self.cnt)
        
        with torch.no_grad():
            _, feat = model(self.ctx, self.cnt)
            print(f"F3 Model Feature Shape: {feat.shape}")
        
        avg_time = self._benchmark_model(feature_test)
        results['feature_extraction'] = avg_time
        print(f"F3 Model Average time: {avg_time:.2f} ms")
        torch.cuda.empty_cache()
        
        flowmodel = EventFFFlow(eventff_config=self.cfg, flowhead_config=self.flowhead_config).cuda()
        flowmodel.eventff = torch.compile(flowmodel.eventff, fullgraph=False)
        flowmodel.flowhead = torch.compile(flowmodel.flowhead, fullgraph=False)
        
        def flow_test():
            flowmodel(self.ctx, self.cnt, cparams=self.cparams)
        
        avg_time = self._benchmark_model(flow_test)
        results['optical_flow'] = avg_time
        print(f"F3 Flow Average time: {avg_time:.2f} ms")
        torch.cuda.empty_cache()
        
        depthmodel = EventFFDepthAnythingV2(self.cfg, self.depth_config).cuda()
        depthmodel.eventff = torch.compile(depthmodel.eventff, fullgraph=False)
        
        def depth_test():
            depthmodel.infer_image(self.ctx, self.cnt, cparams=self.cparams[0])
        
        avg_time = self._benchmark_model(depth_test)
        results['depth_estimation'] = avg_time
        print(f"F3 Depth Average time: {avg_time:.2f} ms")
        torch.cuda.empty_cache()
        
        segmodel = EventFFSegformer(self.cfg, segformer_config=self.seg_config).cuda()
        segmodel.eventff = torch.compile(segmodel.eventff, fullgraph=False)
        
        def seg_test():
            segmodel(self.ctx, self.cnt, cparams=self.cparams)
        
        avg_time = self._benchmark_model(seg_test)
        results['segmentation'] = avg_time
        print(f"F3 Segmentation Average time: {avg_time:.2f} ms")
        torch.cuda.empty_cache()
        
        return results
    
    def test_event_model(self, eventmodel: str = "voxelgrid") -> dict[str, float]:
        """
        Test EventModel performance with different representations.
        
        Args:
            eventmodel: Type of event representation ('voxelgrid' or 'frames')
            
        Returns:
            Dictionary with performance results
        """
        model_name = "V3" if eventmodel == "voxelgrid" else "I3"
        rep_name = "VoxelGrid" if eventmodel == "voxelgrid" else "Frames"
        
        print(f"\n{'='*60}")
        print(f"{model_name} (EventModel - {rep_name}) PERFORMANCE TESTING")
        print("="*60)
        
        results = {}
        
        flowmodel = EventFlow(
            eventmodel=eventmodel,
            width=self.width,
            height=self.height,
            timebins=self.timebins,
            flowhead_config=self.flowhead_config,
        ).cuda()
        flowmodel.upchannel = torch.compile(flowmodel.upchannel, fullgraph=False)
        flowmodel.flowhead = torch.compile(flowmodel.flowhead, fullgraph=False)
        
        def flow_test():
            flowmodel(self.ctx, self.cnt, cparams=self.cparams)
        
        avg_time = self._benchmark_model(flow_test)
        results['optical_flow'] = avg_time
        print(f"{model_name} Flow Average time: {avg_time:.2f} ms")
        torch.cuda.empty_cache()
        
        depthmodel = EventDepthAnythingV2(
            self.depth_config,
            eventmodel=eventmodel,
            width=self.width,
            height=self.height,
            timebins=self.timebins,
        ).cuda()
        
        def depth_test():
            depthmodel.infer_image(self.ctx, self.cnt, cparams=self.cparams[0])
        
        avg_time = self._benchmark_model(depth_test)
        results['depth_estimation'] = avg_time
        print(f"{model_name} Depth Average time: {avg_time:.2f} ms")
        torch.cuda.empty_cache()

        segmodel = EventSegformer(
            segformer_config=self.seg_config,
            eventmodel=eventmodel,
            width=self.width,
            height=self.height,
            timebins=self.timebins,
        ).cuda()
        
        def seg_test():
            segmodel(self.ctx, self.cnt, cparams=self.cparams)
        
        avg_time = self._benchmark_model(seg_test)
        results['segmentation'] = avg_time
        print(f"{model_name} Segmentation Average time: {avg_time:.2f} ms")
        torch.cuda.empty_cache()
        
        return results
    
    def run_all_tests(self) -> dict[str, dict[str, float]]:
        """Run all performance tests and return comprehensive results."""
        all_results = {}
        all_results['eventff'] = self.test_eventff_model()
        all_results['voxelgrid'] = self.test_event_model('voxelgrid')
        all_results['frames'] = self.test_event_model('frames')
        
        print(f"\n{'='*60}")
        print("PERFORMANCE COMPARISON COMPLETE")
        print("="*60)
        
        return all_results


def print_performance_summary(results: dict[str, dict[str, float]]):
    """Print a formatted summary of performance results."""
    print(f"\n{'='*60}")
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    tasks = ['feature_extraction', 'optical_flow', 'depth_estimation', 'segmentation']
    task_names = ['Feature Extraction', 'Optical Flow', 'Depth Estimation', 'Segmentation']
    
    print(f"{'Task':<20} {'EventFF (F3)':<15} {'VoxelGrid (V3)':<15} {'Frames (I3)':<15}")
    print("-" * 70)
    
    for task, name in zip(tasks, task_names):
        eventff_time = results.get('eventff', {}).get(task, 'N/A')
        voxel_time = results.get('voxelgrid', {}).get(task, 'N/A')
        frame_time = results.get('frames', {}).get(task, 'N/A')
        
        eventff_str = f"{eventff_time:.2f} ms" if eventff_time != 'N/A' else 'N/A'
        voxel_str = f"{voxel_time:.2f} ms" if voxel_time != 'N/A' else 'N/A'
        frame_str = f"{frame_time:.2f} ms" if frame_time != 'N/A' else 'N/A'
        
        print(f"{name:<20} {eventff_str:<15} {voxel_str:<15} {frame_str:<15}")


def main():
    """Main function to run performance tests."""
    # Configuration paths
    config_options = [
        "confs/ff/modeloptions/1280x720x20_patchff_ds1_small.yml",
        "confs/ff/modeloptions/640x480x20_patchff_ds1_small.yml",
        "confs/ff/modeloptions/346x260x50_patchff_ds1_small.yml",
    ]

    all_results = {}

    for cfg in config_options:
        print(f"Available Config: {cfg}")
        tester = PerformanceTester(
            config_path=cfg,
            width=1280,
            height=720,
            timebins=20,
            n_events=200000,
            runs=1000
        )        
        results = tester.run_all_tests()
        print_performance_summary(results)
        all_results[cfg] = results

    return all_results


if __name__ == "__main__":
    results = main()
