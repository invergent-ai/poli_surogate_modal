"""
Merge LoRA pe Modal - combină adapter-ul LoRA cu modelul de bază într-un
director gata de servit (se poate încărca direct cu `AutoModelForCausalLM`
din transformers, vLLM etc.).

Folosire:

    modal run scripts/merge_checkpoint.py \\
        --base-model Qwen/Qwen3-0.6B \\
        --checkpoint-dir /output \\
        --output /output/merged

Argumente:
  --base-model       HF id (ex. Qwen/Qwen3-0.6B) sau cale locală
  --checkpoint-dir   director din Volume cu adapter_model.safetensors +
                     adapter_config.json (`/output` e montat din Volume)
  --output           director destinație în Volume (persistă între rulări)

După merge, descarci modelul local cu:

    modal volume get surogate-outputs /merged ./merged

Notă: Dacă vrei ca Surogate să facă merge-ul AUTOMAT la finalul
antrenamentului, adaugă `merge_adapter: true` în YAML. Script-ul ăsta e
util doar pentru checkpoint-uri antrenate fără acel parametru.
"""
import subprocess
import textwrap

import modal

# Python-ul instalat de Surogate în container. Singurul care poate face
# `from surogate.utils.adapter_merge import merge_adapter`.
SUROGATE_PYTHON = "/opt/surogate/.venv/bin/python"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu24.04", add_python="3.12"
    )
    .apt_install("curl", "git", "ca-certificates")
    .run_commands(
        "mkdir -p /opt/surogate",
        "cd /opt/surogate && curl -LsSf https://surogate.ai/install.sh | bash",
        gpu="T4",
    )
)

vol = modal.Volume.from_name("surogate-outputs", create_if_missing=True)
app = modal.App("surogate-merge")

# Codul care face merge-ul efectiv. Rulează cu Python-ul Surogate (are
# acces la `surogate.utils.adapter_merge`). Primește 3 argumente pe argv.
_MERGE_CODE = textwrap.dedent("""
    import os
    import sys

    from surogate.utils.adapter_merge import merge_adapter

    base, ckpt, out = sys.argv[1:4]

    # Dacă baza nu e un director local, o descarcă de pe HuggingFace.
    if not os.path.isdir(base):
        from huggingface_hub import snapshot_download
        print(f"Descarc modelul de bază de pe HF: {base}")
        base = snapshot_download(base)

    # Verifică că checkpoint-ul are fișierele adapter.
    if not os.path.exists(os.path.join(ckpt, "adapter_model.safetensors")):
        sys.exit(f"Eroare: nu am găsit adapter_model.safetensors în {ckpt}")
    if not os.path.exists(os.path.join(ckpt, "adapter_config.json")):
        sys.exit(f"Eroare: nu am găsit adapter_config.json în {ckpt}")

    print(f"Model de bază: {base}")
    print(f"Checkpoint:    {ckpt}")
    print(f"Output:        {out}")

    merge_adapter(base_model_path=base, adapter_path=ckpt, output_path=out)
    print(f"\\nModel combinat salvat în {out}")
""")


@app.function(image=image, gpu="T4", timeout=20 * 60, volumes={"/output": vol})
def merge(base_model: str, checkpoint_dir: str, output: str) -> None:
    subprocess.run(
        [SUROGATE_PYTHON, "-c", _MERGE_CODE, base_model, checkpoint_dir, output],
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
