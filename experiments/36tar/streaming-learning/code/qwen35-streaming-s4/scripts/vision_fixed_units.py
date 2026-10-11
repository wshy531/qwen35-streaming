"""Experimental reversible temporal-unit linear execution, never a default model patch."""
import torch


class FixedTemporalLinears:
    def __init__(self, visual, max_operations=300):
        self.visual = visual
        self.max_operations = max_operations
        self.operations = 0
        self.calls = 0
        self.units = None
        self.saved = []
        self.hook = None

    def grid(self, _module, args, kwargs):
        grid = kwargs.get("grid_thw", args[1] if len(args) > 1 else None)
        if grid is None or tuple(grid.shape) != (1, 3):
            raise ValueError("Single-video grid required")
        t, h, w = (int(v) for v in grid[0])
        merge = self.visual.spatial_merge_size
        if min(t, h, w) < 1 or h % merge or w % merge:
            raise ValueError("Positive aligned grid required")
        pixels = args[0] if args else kwargs["hidden_states"]
        if pixels.shape[0] != t * h * w:
            raise ValueError("Pixel rows/grid mismatch")
        self.units = h * w, h * w // merge**2
        self.calls += 1

    def __enter__(self):
        self.hook = self.visual.register_forward_pre_hook(self.grid, with_kwargs=True)
        for name, module in self.visual.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            existed, previous = "forward" in module.__dict__, module.__dict__.get("forward")
            original = module.forward
            self.saved.append((module, existed, previous))
            def fixed(value, _original=original, _merger=name.startswith("merger.")):
                if self.units is None:
                    raise RuntimeError("No active grid for temporal-unit linear")
                size = self.units[1 if _merger else 0]
                if value.ndim != 2 or value.shape[0] % size:
                    raise ValueError("Linear input rows not aligned to temporal unit")
                operations = value.shape[0] // size
                if self.operations + operations > self.max_operations:
                    raise RuntimeError("Temporal-unit projection budget exceeded")
                self.operations += operations
                if operations == 1:
                    return _original(value)
                return torch.cat([_original(part) for part in value.split(size)], 0)
            module.forward = fixed
        return self

    def __exit__(self, *_exception):
        if self.hook is not None:
            self.hook.remove()
        for module, existed, value in self.saved:
            if existed:
                module.forward = value
            else:
                delattr(module, "forward")
        self.saved.clear()
        self.units = None
