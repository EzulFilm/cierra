"""Run a generated WGSL shader through wgpu-py (wgpu-native) for parity tests. A test harness, not a production
path: a host (a browser, a WebGPU app) binds the same shader to its own buffers.

wgpu picks the adapter: without a GPU that is Mesa's llvmpipe (a CPU Vulkan driver), unless the environment exposes one
(for example GALLIUM_DRIVER=d3d12 for the OpenGL backend through Mesa's D3D12 driver). Nothing is kept between
calls: the device is destroyed after each run.
"""

import numpy as np

_ADAPTER = {}


def adapter():
    """(wgpu adapter, None) or (None, reason): why the harness cannot run here."""
    if "a" not in _ADAPTER:
        try:
            import wgpu
        except ImportError as e:
            _ADAPTER["a"] = (None, f"wgpu not installed ({e})")
        else:
            try:
                a = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
                _ADAPTER["a"] = (a, None) if a is not None else (None, "wgpu found no adapter")
            except Exception as e:  # wgpu-native raises plain RuntimeErrors when no backend initialises
                _ADAPTER["a"] = (None, f"wgpu found no adapter: {type(e).__name__}: {e}")
    return _ADAPTER["a"]


def adapter_name():
    a, why = adapter()
    return why if a is None else f"{a.info['device']} ({a.info['adapter_type']}, {a.info['backend_type']})"


def run_wgsl(wgsl_text, code):
    """Apply a shader from compile_wgsl to ACEScct (3, H, W); returns float64 (3, H, W)."""
    import wgpu
    from .wgsl import WORKGROUP
    a, why = adapter()
    if a is None:
        raise RuntimeError(why)
    _, h, w = code.shape
    data = np.ascontiguousarray(code.transpose(1, 2, 0), dtype=np.float32)
    dev = a.request_device_sync()
    try:
        module = dev.create_shader_module(code=wgsl_text)
        src = dev.create_buffer_with_data(data=data, usage=wgpu.BufferUsage.STORAGE)
        dst = dev.create_buffer(size=data.nbytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
        size = dev.create_buffer_with_data(data=np.array([w, h], dtype=np.uint32), usage=wgpu.BufferUsage.UNIFORM)
        pipe = dev.create_compute_pipeline(layout="auto", compute={"module": module, "entry_point": "main"})
        group = dev.create_bind_group(layout=pipe.get_bind_group_layout(0), entries=[
            {"binding": 0, "resource": {"buffer": src}}, {"binding": 1, "resource": {"buffer": dst}},
            {"binding": 2, "resource": {"buffer": size}}])
        enc = dev.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(pipe)
        cp.set_bind_group(0, group)
        cp.dispatch_workgroups(-(-w // WORKGROUP[0]), -(-h // WORKGROUP[1]), 1)
        cp.end()
        dev.queue.submit([enc.finish()])
        out = np.frombuffer(dev.queue.read_buffer(dst), dtype=np.float32).reshape(h, w, 3)
        for b in (src, dst, size):
            b.destroy()
    finally:
        dev.destroy()
    return out.transpose(2, 0, 1).astype(np.float64)
