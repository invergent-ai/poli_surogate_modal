# Surogate SFT pe Modal

Ghid practic pentru antrenare (SFT - Supervised Fine-Tuning) cu
[Surogate](https://docs.surogate.ai) pe GPU închiriat de la
[Modal](https://modal.com). Include 4 rulări gata făcute pe 4 GPU-uri
diferite, cu dataset-ul românesc `OpenLLM-Ro/ro_gsm8k`.

**Structura repo-ului:**

| Fișier                        | Rol                                                  |
| ----------------------------- | ---------------------------------------------------- |
| `run_gpu_test.py`             | Script Modal cu 4 funcții, una pe GPU - rulează SFT  |
| `configs/bf16.yaml`           | Config SFT recipe **bf16** (merge pe orice GPU)      |
| `configs/fp8.yaml`            | Config SFT recipe **fp8-hybrid** (Ada+)              |
| `configs/nvfp4.yaml`          | Config SFT recipe **nvfp4** (doar Blackwell)         |
| `configs/chess_pure.yaml`     | Exemplu dataset HF cu coloane `input`/`output`       |
| `configs/chess_mix.yaml`      | Exemplu dataset HF în format chat (`messages`)       |
| `scripts/merge_checkpoint.py` | Script Modal care combină LoRA-ul cu modelul de bază |
| `images/`                     | Screenshot-uri din Modal de la rulările de referință |

---

## 1. Setup

**Pasul 0 - cont Modal.** Înainte de orice, intră pe
[modal.com](https://modal.com) și fă-ți cont (gratuit, iți dă $30 credit
lunar). Fără cont, `modal token new` nu are unde să te autentifice.

**Pasul 1 - venv + modal.** Din folderul repo-ului:

```bash
uv venv --python 3.12                # creează .venv/ cu Python 3.12
source .venv/bin/activate            # activează mediul virtual
uv pip install modal                 # instalează Modal CLI + SDK
```

> Dacă n-ai `uv`, îl instalezi cu `curl -LsSf https://astral.sh/uv/install.sh | sh`.
> Alternativ, poți folosi `python3.12 -m venv .venv && pip install modal`.
>
> **Pe Linux**, după instalare adaugă `uv` în PATH cu `source $HOME/.local/bin/env`
> (sau deschide un terminal nou). Pentru permanent: `echo 'source $HOME/.local/bin/env' >> ~/.bashrc`.

**Pasul 2 - autentificare.** Încearcă:

```bash
modal setup # deschide browserul, te loghezi, salvează token-ul
```

Dacă din vreun motiv `modal` nu e pe PATH, rulează varianta echivalentă:

```bash
python -m modal setup
```

---

## 2. Alege GPU-ul

Modal nu oferă RTX 5090. Tabelul complet al GPU-urilor și ce recipe
Surogate suportă fiecare:

| GPU Modal     | VRAM   | bf16 | fp8-hybrid | nvfp4 | $/oră |
| ------------- | ------ | ---- | ---------- | ----- | ----- |
| T4            | 16 GB  | ✓    | ✗          | ✗     | 0.59  |
| L4            | 24 GB  | ✓    | ✓          | ✗     | 0.80  |
| A10           | 24 GB  | ✓    | ✗          | ✗     | 1.10  |
| L40S          | 48 GB  | ✓    | ✓          | ✗     | 1.95  |
| A100-40GB     | 40 GB  | ✓    | ✗          | ✗     | 2.10  |
| **A100-80GB** | 80 GB  | ✓    | ✗          | ✗     | 2.50  |
| RTX PRO 6000  | 96 GB  | ✓    | ✓          | ✓     | 3.03  |
| **H100**      | 80 GB  | ✓    | ✓          | ✗     | 3.95  |
| H200          | 141 GB | ✓    | ✓          | ✗     | 4.54  |
| B200          | 180 GB | ✓    | ✓          | ✓     | 6.25  |

Multi-GPU: adaugă `:N` la string (ex. `"H100:2"` pentru 2× H100).

**Recomandări scurte pentru SFT:**

- **Buget** → `L4` (bf16) sau `L40S` (fp8-hybrid).
- **Default solid** → `A100-80GB` (bf16) sau `H100` (fp8-hybrid).
- **Modele mari / context lung** → `H200`, `B200`.

---

## 3. Rulează o rulare de referință

Cele 4 entry-points din `run_gpu_test.py` au fost rulate și verificate
end-to-end. Le pornești individual:

```bash
modal run run_gpu_test.py::test_l4              # L4  + bf16    (~2.5 min)
modal run run_gpu_test.py::test_l40s            # L40S + fp8    (~1 min)
modal run run_gpu_test.py::test_a100            # A100 + bf16   (~1 min)
modal run run_gpu_test.py::test_rtx_pro_6000    # RTX PRO + nvfp4 (~45 s)
```

Prima rulare construiește imaginea Docker în Modal (~3 min, se
cache-uiește). Următoarele rulări pornesc containerul în ~10 s.

**Cum arată când rulează în Modal** (UI-ul Modal, tab `Logs`):

![Training în curs](images/gpu_surogate_poli1.png)

---

## 4. Rezultate ale celor 4 rulări (Qwen3-0.6B + LoRA, 50 pași)

| #   | GPU          | Config                                     | Timp  | Throughput  | Loss start → final |
| --- | ------------ | ------------------------------------------ | ----- | ----------- | ------------------ |
| 1   | L4           | [`configs/bf16.yaml`](configs/bf16.yaml)   | 156 s | 6.7k tok/s  | 1.78 → 0.83        |
| 2   | L40S         | [`configs/fp8.yaml`](configs/fp8.yaml)     | 61 s  | 25.4k tok/s | 1.81 → 0.84        |
| 3   | A100-80GB    | [`configs/bf16.yaml`](configs/bf16.yaml)   | 59 s  | 22.0k tok/s | 1.78 → 0.83        |
| 4   | RTX PRO 6000 | [`configs/nvfp4.yaml`](configs/nvfp4.yaml) | 45 s  | 39.6k tok/s | 1.85 → 0.89        |

### Detalii scurte pe fiecare

**1. L4 + bf16** - cea mai ieftină opțiune. Throughput redus dar loss
identic cu GPU-urile mari (bf16 e determinist). Bun pentru verificări.

**2. L40S + fp8-hybrid** - sweet spot calitate/preț: peak BF16 362 TFLOP/s,
peak FP8 733 TFLOP/s. ~2.5× mai rapid decât L4 pentru același rezultat.

**3. A100-80GB + bf16** - aceeași viteză ca L40S, dar 80 GB VRAM → poți
antrena modele mult mai mari (LoRA pe ~30B sau full fine-tune pe ~3B).

**4. RTX PRO 6000 + nvfp4** - cea mai rapidă (39.6k tok/s). Pipeline-ul
CUTLASS FP4 e primat automat ("FP4 cache primed: 96 fwd + 96 bwd").
Loss puțin mai mare - compromis așteptat pentru precizie 4-bit.

**Cum arată la final** (LoRA salvat, training complete):

![Training finalizat](images/gpu_surogate_poli2.png)

---

## 5. Descarcă rezultatele

```bash
modal volume ls surogate-outputs              # listează conținutul volumului
modal volume get surogate-outputs / ./out     # descarcă-l local în ./out/
```

Ce găsești în volum după un run:

| Fișier                            | Rol                                     |
| --------------------------------- | --------------------------------------- |
| `adapter_model.safetensors`       | Weights-urile LoRA (adapterul antrenat) |
| `adapter_config.json`             | Meta-date LoRA (rank, alpha, etc.)      |
| `log-<nume-run>-<timestamp>.json` | Log complet cu toți pașii               |
| `training_plot.png`               | Graficul cu loss-ul                     |
| `train-000.bin`, `eval-000.bin`   | Dataset tokenizat (cache)               |
| `.tokenize_hash`                  | Hash pentru invalidare cache            |

> ⚠️ **Atenție:** toate rulările scriu în același `output_dir`, deci
> `adapter_model.safetensors` se **suprascrie** la fiecare run. Ca să
> păstrezi mai multe checkpoint-uri, schimbă `output_dir` în YAML
> (ex. `/output/l40s-fp8-run1`) sau folosește volume separate.

---

## 6. Merge LoRA - obține modelul gata de servit

După antrenament ai doar adapter-ul LoRA (~20 MB). Ca să-l folosești cu
`AutoModelForCausalLM` (transformers, vLLM, etc.) trebuie să-l combini
cu modelul de bază.

### Varianta recomandată - automat, din YAML

Surogate face merge-ul **la sfârșitul antrenamentului** dacă adaugi în
config:

```yaml
merge_adapter: true
```

(default e `false`). Când e activ, după ultimul pas Surogate salvează în
`output_dir` **și** adapter-ul LoRA **și** modelul combinat - deci nu mai
ai nevoie de script extern.

### Varianta manuală - când NU ai setat `merge_adapter`

Dacă ai antrenat fără `merge_adapter: true` (sau vrei să combini un
checkpoint vechi cu alt model de bază), `scripts/merge_checkpoint.py`
face merge-ul pe Modal, folosind Python-ul Surogate din container (nu
trebuie instalat nimic local).

```bash
modal run scripts/merge_checkpoint.py \
    --base-model Qwen/Qwen3-0.6B \
    --checkpoint-dir /output \
    --output /output/merged
```

Argumente:

| Argument           | Ce e                                                                   |
| ------------------ | ---------------------------------------------------------------------- |
| `--base-model`     | HF id (`Qwen/Qwen3-0.6B`) sau un director local cu modelul de bază     |
| `--checkpoint-dir` | Director cu `adapter_model.safetensors` + `adapter_config.json`        |
| `--output`         | Unde se scrie modelul combinat (tot în Volume → persistă între rulări) |

După merge, descarci modelul local:

```bash
modal volume get surogate-outputs /merged ./merged
```

**Intern:** Python-ul Modal nu are acces la `surogate` (e instalat într-un
venv separat, `/opt/surogate/.venv/`). Script-ul apelează acel Python cu
`subprocess`, care face `from surogate.utils.adapter_merge import
merge_adapter` și combină weights-urile LoRA direct în straturile modelului.

---

## 7. Parametrii cei mai importanți din YAML (scurt)

Pentru detalii complete vezi
[documentația Surogate](https://docs.surogate.ai/guides/configuration).

| Parametru                     | Ce face                                                                             |
| ----------------------------- | ----------------------------------------------------------------------------------- |
| `model`                       | HF id al modelului de bază, ex. `Qwen/Qwen3-0.6B`, `meta-llama/Llama-3.1-8B`        |
| `datasets[].path`             | HF id sau cale locală `.jsonl`; `type: auto` detectează formatul                    |
| `recipe`                      | `bf16` (orice GPU) / `fp8-hybrid` (Ada+) / `nvfp4` (Blackwell)                      |
| `max_steps`                   | Număr total de pași. 50 = smoke test; 500-5000 = antrenament real                   |
| `per_device_train_batch_size` | Exemple per GPU per pas. Crește dacă ai VRAM, scade dacă OOM                        |
| `gradient_accumulation_steps` | Acumulează gradienți peste N mini-batches (batch efectiv = batch × accum)           |
| `sequence_len`                | Lungime maximă în tokeni. 2048 standard; 4096/8192 pentru context lung, crește VRAM |
| `sample_packing`              | `true` = împachetează documente în aceeași secvență (mult mai eficient)             |
| `learning_rate`               | LoRA: `2e-4`. Full fine-tune: `1e-5 … 5e-5`. Prea mare = divergență                 |
| `warmup_ratio`                | Procent din pași cu LR crescând liniar de la 0. Tipic 0.05–0.15                     |
| `lora`                        | `true` = antrenează adaptori peste model înghețat (rapid, VRAM mic)                 |
| `lora_rank`                   | Dimensiunea matricilor LoRA. Tipic 8–64. Mai mare = mai multă capacitate + VRAM     |
| `lora_alpha`                  | Factor de scalare (convenție: `2 * lora_rank`)                                      |
| `lora_target_modules`         | Ce straturi primesc adaptori. Lista din configuri = "all-linear" pentru Qwen        |

**LoRA pe scurt:** înghețăm modelul de bază, antrenăm doar niște matrici
mici (low-rank) care se adaugă la atenție/MLP. Economisim ~90% VRAM față
de full fine-tune, cu rezultate foarte apropiate pentru adaptare pe
domenii specifice.

---

## 8. Modificări tipice

- **Alt dataset:** în YAML, `datasets[0].path` → HF id (ex.
  `tatsu-lab/alpaca`) sau cale `/workspace/my.jsonl`. Dacă folosești JSONL
  local, adaugă-l în imagine cu
  `image.add_local_file("data.jsonl", "/workspace/data.jsonl")`.
  Pentru detalii pe formatele de dataset (instruction vs conversation,
  validation split, system prompt) vezi secțiunea **8.1**.
- **Alt model:** `model: <HF_ID>`. Pentru modele mai mari (7B+) verifică
  că GPU-ul are VRAM suficient; scade `per_device_train_batch_size` dacă e
  cazul.
- **Mai mulți pași:** `max_steps: 1000` (sau șterge linia pentru 1 epocă
  completă).
- **Altă precizie:** schimbă `recipe:` și alege GPU din tabelul de la
  secțiunea 2.
- **Alt GPU:** adaugă o funcție nouă în `run_gpu_test.py`:
  ```python
  @app.function(gpu="H100", **_fn_kwargs)
  def test_h100():
      _train("/workspace/fp8.yaml", "H100 + fp8-hybrid")
  ```
  Apoi `modal run run_gpu_test.py::test_h100`.

---

## 8.1. Exemplu real - două dataset-uri de șah pe HF

Două situații tipice cu dataset-uri de pe HuggingFace, cu config-uri gata
făcute în `configs/chess_pure.yaml` și `configs/chess_mix.yaml`. Diferența
între ele e **schema rândurilor**, nu conținutul.

### Caz A - dataset cu coloane `input`/`output` (instruction)

[`cetusian/chess-sft-lichess-2200`](https://huggingface.co/datasets/cetusian/chess-sft-lichess-2200)
are pe fiecare rând două coloane string: `input` (poziția + istoricul
mutărilor) și `output` (mutarea bună în SAN). Splituri: `train` +
`validation`.

Surogate nu poate folosi direct astfel de rânduri - trebuie să-i spui ce
coloană devine user turn și ce coloană devine assistant turn. Asta face
`type: instruction`:

```yaml
datasets:
  - path: cetusian/chess-sft-lichess-2200   # HF repo id
    split: train
    type: instruction                       # input+output → chat
    instruction_field: input                # coloana → user turn
    output_field: output                    # coloana → assistant turn
    system_prompt_type: fixed               # un singur system prompt fix
    system_prompt: "You are a chess grandmaster. Given a position and move history, respond with the best next move in SAN notation."

validation_datasets:
  - path: cetusian/chess-sft-lichess-2200
    split: validation                       # alt split, restul identic
    type: instruction
    instruction_field: input
    output_field: output
    system_prompt_type: fixed
    system_prompt: "You are a chess grandmaster. Given a position and move history, respond with the best next move in SAN notation."
```

Rulezi cu (folosind același entry-point Modal, doar schimbi YAML-ul în
`run_gpu_test.py` sau adaugi o funcție nouă):

```bash
modal run run_gpu_test.py::test_a100   # după ce point-ezi la chess_pure.yaml
```

### Caz B - dataset deja în format chat (conversation)

[`cetusian/chess-sft-mix-200k`](https://huggingface.co/datasets/cetusian/chess-sft-mix-200k)
are o singură coloană, `messages`, care e deja o listă de
`{"role": ..., "content": ...}` (system + user + assistant). Tot
`train` + `validation`.

Aici Surogate nu trebuie să transforme nimic - îi spui doar că rândurile
sunt deja conversații cu `type: conversation`:

```yaml
datasets:
  - path: cetusian/chess-sft-mix-200k
    split: train
    type: conversation
    messages_field: messages                # default e tot "messages", explicit pentru claritate

validation_datasets:
  - path: cetusian/chess-sft-mix-200k
    split: validation
    type: conversation
    messages_field: messages
```

System prompt-ul vine din `messages[0]` dacă există în date - nu mai
setezi `system_prompt` separat (ar fi ignorat).

### Cum aleg între cele două

| Dacă rândul tău arată ca...                                  | Folosește           |
| ------------------------------------------------------------ | ------------------- |
| `{"input": "...", "output": "..."}` (sau alte 2 coloane string) | `type: instruction` cu `instruction_field` + `output_field` |
| `{"messages": [{"role": "...", "content": "..."}, ...]}`     | `type: conversation` cu `messages_field`                    |
| Format alpaca standard (`instruction`/`input`/`output`)      | `type: alpaca` (Surogate îl recunoaște direct)              |
| Nu ești sigur                                                | `type: auto` - lasă Surogate să detecteze                   |

### Două note importante

1. **`validation_datasets:` cere `eval_steps > 0`** în config. Dacă lași
   `eval_steps: 0` (cum e în `bf16.yaml`), val loss-ul nu se calculează
   chiar dacă declari split-ul. Ambele config-uri chess au `eval_steps: 50`.
2. **`max_steps: 50`** din config-urile de smoke-test e prea mic pentru un
   antrenament real pe 200k exemple. `chess_pure.yaml` și `chess_mix.yaml`
   au `max_steps: 200` - tot doar pentru a vedea că merge end-to-end.
   Pentru rulare reală pe tot dataset-ul, șterge linia (`= 1 epocă`) sau
   pune o valoare mare (ex. 5000).

---

## 8.2. Token HF - când e nevoie și cum se setează

**Pentru cele două dataset-uri din 8.1 (`cetusian/chess-sft-*`) NU îți
trebuie token** - sunt publice (CC0), Modal le descarcă anonim. La fel
pentru `OpenLLM-Ro/ro_gsm8k` și `Qwen/Qwen3-0.6B`.

**Ai nevoie de token doar dacă** modelul sau dataset-ul tău e:

- gated (ex. `meta-llama/Llama-3.1-8B`, `google/gemma-2-9b`, dataset-uri
  cu "Agree to share contact info"),
- privat (repo-uri din contul/organizația ta neexpuse public).

### Pasul 1 - obține token-ul

[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
→ **New token** → tip `Read` (suficient pentru download). Pentru modele
gated, intră înainte pe pagina modelului și apasă "Agree and access".

### Pasul 2 - creează un Modal Secret cu el

```bash
modal secret create huggingface HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
```

(Numele `huggingface` e doar o convenție - poți folosi orice. Verifici
cu `modal secret list`.)

### Pasul 3 - atașează secret-ul la funcția de antrenament

În `run_gpu_test.py`, adaugă `secrets=[...]` la decorator. Cea mai
simplă variantă - îl pui în `_fn_kwargs` ca să se aplice la toate
funcțiile dintr-o dată:

```python
_fn_kwargs = dict(
    image=image,
    timeout=30 * 60,
    volumes={"/output": vol},
    secrets=[modal.Secret.from_name("huggingface")],   # NOU
)
```

Modal expune `HF_TOKEN` ca variabilă de mediu în container, iar
`huggingface_hub` (folosit intern de Surogate) o detectează automat -
nu trebuie cod în plus.

### Verificare rapidă

Dacă token-ul lipsește pentru un repo gated, eroarea în log e:

```
huggingface_hub.utils._errors.GatedRepoError: 401 Client Error
Cannot access gated repo for url ...
```

Dacă vezi asta cu un dataset/model care ar trebui să fie public, e
probabil typo în `path:` din YAML (HF returnează tot 401 pentru
repo-uri inexistente, nu 404).

---

## 9. Debugging - probleme întâlnite și soluții

Au apărut în timpul validării acestui ghid - dacă le vezi, iată fix-urile:

| Eroare                                                      | Cauză                                              | Fix                                                                         |
| ----------------------------------------------------------- | -------------------------------------------------- | --------------------------------------------------------------------------- |
| `wheel is compatible with manylinux_2_39, you're on 2_35`   | Base image Ubuntu 22.04 (glibc prea vechi)         | Folosește `nvidia/cuda:12.8.0-devel-ubuntu24.04`                            |
| `curl is required but not installed` (deși curl e prezent)  | Installer-ul e bash, dar l-ai pipe-uit la `sh`     | Schimbă `\| sh` → `\| bash`                                                 |
| `ModuleNotFoundError: No module named 'grpclib'` la runtime | Ai pus venv-ul Surogate pe PATH → ascunde Modal    | Nu suprascrie PATH; apelează `/opt/surogate/.venv/bin/surogate` direct      |
| `image tried to run a build step after add_local_*`         | `run_commands` sau `workdir` după `add_local_file` | Pune `add_local_file(...)` la SFÂRȘITUL pipeline-ului de build              |
| `CUDA out of memory` în timpul antrenamentului              | Batch prea mare sau `sequence_len` prea lung       | Scade `per_device_train_batch_size` sau `sequence_len`; crește `grad_accum` |
| Installer-ul Surogate cere CUDA la build                    | Modal build fără GPU nu detectează CUDA            | Pune `gpu="T4"` pe `run_commands` unde rulezi installer-ul                  |

---

## 10. Tips

- **Cache-ul imaginii** - Modal rebuildește doar ce s-a schimbat. Dacă
  editezi doar YAML-urile, se recompune doar layer-ul `add_local_file`
  (secunde). Dacă schimbi `run_commands`, se face build complet (~3 min).
- **Cache-ul dataset-ului** - tokenizarea `OpenLLM-Ro/ro_gsm8k` se salvează
  în volum după prima rulare. Rulările ulterioare o reutilizează (log:
  _"Tokenization hash unchanged ..."_).
- **Loss determinist** - rulările bf16 pe L4 și A100 dau EXACT același
  loss per pas (seed fix, operații deterministe). Util pentru debugging.
- **Cost real** - în acest repo toate 4 rulările împreună au costat mai
  puțin de $0.30. Nu ezita să experimentezi.
- **Observă `SOL`** - "Speed of Light" din log e procent din peak-ul
  teoretic al GPU-ului. >20% e sănătos pentru modele mici; <5% = problemă.

---

## 11. Referințe

- Documentația Surogate: https://docs.surogate.ai
- Exemple SFT oficiale:
  https://github.com/invergent-ai/surogate/tree/main/examples/sft
- Documentația Modal (GPU, Volumes, Images):
  https://modal.com/docs
- Dataset folosit: https://huggingface.co/datasets/OpenLLM-Ro/ro_gsm8k
