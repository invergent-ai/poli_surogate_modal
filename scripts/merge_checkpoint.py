#!/usr/bin/env python3
"""
Combină un checkpoint LoRA cu modelul de bază și produce un director de model
gata de servit (care poate fi încărcat direct cu `AutoModelForCausalLM`).

Cum îl rulezi:

    python merge_checkpoint.py \\
        --base-model Qwen/Qwen3-0.6B \\
        --checkpoint-dir ./output/step_00000050 \\
        --output ./merged_model

Script-ul trebuie rulat cu Python-ul care are Surogate instalat
(adică venv-ul creat de installer-ul Surogate). În container-ul Modal
din acest repo: `/opt/surogate/.venv/bin/python`.
"""

import argparse
import os
import sys

# Permite rularea din rădăcina repo-ului fără `pip install -e .`
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from surogate.utils.adapter_merge import merge_adapter


def main():
    parser = argparse.ArgumentParser(
        description="Combină un checkpoint LoRA cu modelul de bază."
    )
    parser.add_argument(
        "--base-model", required=True,
        help="Calea către modelul de bază (director local sau HuggingFace model ID).",
    )
    parser.add_argument(
        "--checkpoint-dir", required=True,
        help="Calea către directorul checkpoint-ului (ex. output/step_00000050).",
    )
    parser.add_argument(
        "--output", required=True,
        help="Directorul de ieșire unde va fi salvat modelul combinat.",
    )
    args = parser.parse_args()

    # Rezolvă calea modelului de bază — dacă nu e un director local, descarcă de pe HuggingFace
    base_model_path = args.base_model
    if not os.path.isdir(base_model_path):
        try:
            from huggingface_hub import snapshot_download
            print(f"Descarc modelul de bază de pe HuggingFace: {base_model_path}")
            base_model_path = snapshot_download(base_model_path)
        except Exception as e:
            print(
                f"Eroare: '{args.base_model}' nu e un director local și "
                f"nu a putut fi descărcat de pe HuggingFace: {e}",
                file=sys.stderr,
            )
            sys.exit(1)

    # Validează că directorul checkpoint-ului conține fișierele adapter LoRA
    checkpoint_dir = args.checkpoint_dir
    adapter_file = os.path.join(checkpoint_dir, "adapter_model.safetensors")
    adapter_config = os.path.join(checkpoint_dir, "adapter_config.json")
    if not os.path.exists(adapter_file):
        print(
            f"Eroare: nu am găsit adapter_model.safetensors în {checkpoint_dir}. "
            "Ești sigur că e un checkpoint LoRA?",
            file=sys.stderr,
        )
        sys.exit(1)
    if not os.path.exists(adapter_config):
        print(
            f"Eroare: nu am găsit adapter_config.json în {checkpoint_dir}.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Model de bază:  {base_model_path}")
    print(f"Checkpoint:     {checkpoint_dir}")
    print(f"Output:         {args.output}")

    # Apelul efectiv — Surogate face merge-ul greutăților LoRA în modelul de bază
    merge_adapter(
        base_model_path=base_model_path,
        adapter_path=checkpoint_dir,
        output_path=args.output,
    )

    print(f"\nModel combinat salvat în {args.output}")


if __name__ == "__main__":
    main()
