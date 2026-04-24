"""
Surogate SFT pe Modal — 4 entry-points, una pe GPU.

Folosire:
    modal run run_gpu_test.py::test_l4              # L4  + bf16
    modal run run_gpu_test.py::test_l40s            # L40S + fp8-hybrid
    modal run run_gpu_test.py::test_a100            # A100-80GB + bf16
    modal run run_gpu_test.py::test_rtx_pro_6000    # RTX PRO 6000 + nvfp4

Cum adaugi un GPU nou: copiază una din funcțiile de mai jos, schimbă
`gpu="..."` (vezi tabelul din README) și YAML-ul corespunzător.
"""
import subprocess
import time

import modal

# Binarul Surogate este instalat într-un venv creat de installer.
# NU-l pune pe PATH — ar ascunde Python-ul Modal și ar rupe runtime-ul.
SUROGATE = "/opt/surogate/.venv/bin/surogate"

image = (
    # Ubuntu 24.04 e obligatoriu — wheel-ul Surogate e manylinux_2_39.
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu24.04", add_python="3.12"
    )
    .apt_install("curl", "git", "ca-certificates")
    .run_commands(
        "mkdir -p /opt/surogate /workspace",
        # Pipe-uie la `bash`, NU la `sh` — scriptul folosește bashisms.
        # `gpu="T4"` e necesar ca installer-ul să detecteze CUDA la build.
        "cd /opt/surogate && curl -LsSf https://surogate.ai/install.sh | bash",
        gpu="T4",
    )
    .workdir("/workspace")
    # IMPORTANT: add_local_file trebuie să fie UlTIMUL pas (Modal-cerință).
    .add_local_file("configs/bf16.yaml", "/workspace/bf16.yaml")
    .add_local_file("configs/fp8.yaml", "/workspace/fp8.yaml")
    .add_local_file("configs/nvfp4.yaml", "/workspace/nvfp4.yaml")
)

# Volum persistent — aici rămân LoRA adapter, log-urile și dataset-ul tokenizat.
vol = modal.Volume.from_name("surogate-outputs", create_if_missing=True)

app = modal.App("surogate-gpu-matrix")


def _train(config_path: str, label: str) -> None:
    """Rulează `surogate sft <config>` și raportează timpul."""
    print(f"\n=== [{label}] starting: {config_path} ===")
    print(
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"]
        ).decode()
    )
    t0 = time.time()
    subprocess.run([SUROGATE, "sft", config_path], check=True)
    print(f"\n=== [{label}] finished in {time.time() - t0:.1f}s ===")
    vol.commit()   # forțează flush-ul scrierilor pe Volume înainte de exit


# timeout = 30 min. Crește pentru antrenamente mai lungi.
_fn_kwargs = dict(image=image, timeout=30 * 60, volumes={"/output": vol})


@app.function(gpu="L4", **_fn_kwargs)
def test_l4():
    _train("/workspace/bf16.yaml", "L4 + bf16")


@app.function(gpu="L40S", **_fn_kwargs)
def test_l40s():
    _train("/workspace/fp8.yaml", "L40S + fp8-hybrid")


@app.function(gpu="A100-80GB", **_fn_kwargs)
def test_a100():
    _train("/workspace/bf16.yaml", "A100-80GB + bf16")


@app.function(gpu="RTX-PRO-6000", **_fn_kwargs)
def test_rtx_pro_6000():
    _train("/workspace/nvfp4.yaml", "RTX PRO 6000 + nvfp4")
