import os
from setuptools import setup
from torch.cuda import (is_available as cuda_is_available,
                        current_device as cuda_current_device,
                        get_device_capability as cuda_get_device_capability)
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

nvcc_args = [
    "-O3",
    "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/"),
]

if cuda_is_available():
    try:
        device_id = cuda_current_device()
        compute_capability = cuda_get_device_capability(device_id)
        sm_version = "".join(map(str, compute_capability))
        nvcc_args.append(f"-gencode=arch=compute_{sm_version},code=sm_{sm_version}")
    except Exception as e:
        raise RuntimeError(f"Failed during GPU architecture detection: {e}.")
else:
    sm_versions = [
        "75",  # Turing (GTX 16-series, RTX 20-series, Tesla T4)
        "80",  # Ampere (A100)
        "86",  # Ampere (RTX 30-series)
        "89",  # Ada Lovelace (RTX 40-series, L4, L40)
        "90",  # Hopper (H100, H200)
        "100",  # Blackwell (B100)
        "101",  # Blackwell (B200)
    ]
    arch_keys = [f"-gencode=arch=compute_{sm_version},code=sm_{sm_version}" for sm_version in sm_versions]
    nvcc_args.extend(arch_keys)

setup(
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
                "cuda_rasterizer/rasterizer_impl.cu",
                "cuda_rasterizer/forward.cu",
                "cuda_rasterizer/backward.cu",
                "rasterize_points.cu",
                "ext.cpp",
            ],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': nvcc_args},
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
