"""
Urcă modelul merged din Volume direct pe HuggingFace Hub. Rulează pe
Modal, deci nu trebuie să descarci modelul local înainte de upload (un
8B fp16 are ~16 GB, e dureros pe net rezidențial).

Folosire:

    modal run scripts/push_to_hf.py --repo-id username/numele-modelului

Argumente:
  --repo-id           Repo destinație pe HF (ex. madalin/llama31-chess)
  --source            Director din Volume (default: /output/merged)
  --private           Dacă repo-ul să fie privat (default: False)
  --commit-message    Mesaj la upload (default: "Push merged model")

Cerințe:
- Modal Secret numit "huggingface" cu HF_TOKEN. Creează-l o singură dată:

      modal secret create huggingface HF_TOKEN=hf_...

  Token-ul cu permisiune "write" îl iei de pe
  https://huggingface.co/settings/tokens
"""
import os

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub>=0.24")
)

vol = modal.Volume.from_name("surogate-outputs", create_if_missing=False)
app = modal.App("surogate-push-hf")


@app.function(
    image=image,
    timeout=60 * 60,
    volumes={"/output": vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def push(
    source: str,
    repo_id: str,
    private: bool,
    commit_message: str,
) -> None:
    from huggingface_hub import HfApi

    if not os.path.isdir(source):
        raise FileNotFoundError(f"Sursa nu există: {source}")

    if not os.path.exists(os.path.join(source, "config.json")):
        raise FileNotFoundError(
            f"Nu am găsit config.json în {source}. "
            f"E sigur că asta e directorul cu modelul merged?"
        )

    # Llama 3.1 Instruct e inutilizabil fără chat template - dacă lipsește
    # tokenizer-ul, urcăm dar avertizăm utilizatorul.
    if not os.path.exists(os.path.join(source, "tokenizer.json")):
        print(
            f"AVERTISMENT: tokenizer.json lipsește din {source}. "
            f"Modelul va fi inutilizabil pentru inferență fără el."
        )

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    print(f"Upload {source} -> https://huggingface.co/{repo_id}")
    api.upload_folder(
        folder_path=source,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )
    print(f"Gata: https://huggingface.co/{repo_id}")


@app.local_entrypoint()
def main(
    repo_id: str,
    source: str = "/output/merged",
    private: bool = False,
    commit_message: str = "Push merged model",
) -> None:
    push.remote(source, repo_id, private, commit_message)
