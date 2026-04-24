"""
Merge LoRA pe Modal — combină adapter-ul LoRA din Volume cu modelul de bază
folosind Python-ul Surogate (cel din container), fără să instalezi nimic
pe laptop.

Folosire:

    modal run merge_lora.py \\
        --base-model Qwen/Qwen3-0.6B \\
        --checkpoint-dir /output \\
        --output /output/merged

Unde:
  --base-model       HF id (ex. Qwen/Qwen3-0.6B) sau director local
  --checkpoint-dir   director cu adapter_model.safetensors + adapter_config.json
                     (în container, `/output` e Volume-ul "surogate-outputs")
  --output           director destinație (tot în Volume → rămâne persistent)

După merge, descarcă modelul local cu:

    modal volume get surogate-outputs /merged ./merged
"""
import subprocess

import modal

SUROGATE_PYTHON = "/opt/surogate/.venv/bin/python"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu24.04", add_python="3.12"
    )
    .apt_install("curl", "git", "ca-certificates")
    .run_commands(
        "mkdir -p /opt/surogate /workspace",
        "cd /opt/surogate && curl -LsSf https://surogate.ai/install.sh | bash",
        gpu="T4",
    )
    .workdir("/workspace")
    .add_local_file("scripts/merge_checkpoint.py", "/workspace/merge_checkpoint.py")
)

vol = modal.Volume.from_name("surogate-outputs", create_if_missing=True)
app = modal.App("surogate-merge")


# Merge-ul e CPU-heavy (încarcă și salvează tensori) — GPU-ul nu e obligatoriu,
# dar un T4 ieftin grăbește serializarea și îți permite verificări rapide.
@app.function(image=image, gpu="T4", timeout=20 * 60, volumes={"/output": vol})
def merge(base_model: str, checkpoint_dir: str, output: str) -> None:
    subprocess.run(
        [
            SUROGATE_PYTHON,
            "/workspace/merge_checkpoint.py",
            "--base-model", base_model,
            "--checkpoint-dir", checkpoint_dir,
            "--output", output,
        ],
        check=True,
    )
    vol.commit()


@app.local_entrypoint()
def main(
    base_model: str = "Qwen/Qwen3-0.6B",
    checkpoint_dir: str = "/output",
    output: str = "/output/merged",
) -> None:
    merge.remote(base_model, checkpoint_dir, output)
