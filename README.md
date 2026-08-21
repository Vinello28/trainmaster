# trainmaster — fine-tuning e validazione automatizzati di piccoli Vision-LLM

[![CI](https://github.com/Vinello28/trainmaster/actions/workflows/ci.yml/badge.svg)](https://github.com/Vinello28/trainmaster/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/uv-package%20manager-DE5FE9)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2013.2-EE4C2C?logo=pytorch&logoColor=white)
![Unsloth](https://img.shields.io/badge/Unsloth-fine--tuning-F97316)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Hub-FFD21E)
![pytest](https://img.shields.io/badge/pytest-tested-0A9EDC?logo=pytest&logoColor=white)

Automatizza il fine-tuning (via [Unsloth](https://unsloth.ai/)) e la validazione di
piccoli Vision-LLM open source su un task di estrazione documentale immagine → JSON
strutturato. Modello target: [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B),
un VLM nativo unificato (early-fusion testo+visione in un'unica architettura).

## Caratteristiche

- **Fine-tuning LoRA o full**, via Unsloth `FastLanguageModel` — un `--set` per passare
  dall'uno all'altro.
- **Validazione semantica**, non solo loss: confronto JSON campo-per-campo, con
  strategia di scoring diversa per tipo di documento (Strategy + registry).
- **Report HTML** predizioni vs ground truth, immagini incluse, ordinato per errore.
- **Testabile senza GPU**: `config`/`data`/`scoring`/`report`/`pipeline` non importano
  mai `torch`/`unsloth` — l'intera suite gira su CI standard.
- **Config YAML riproducibili**, override puntuali da CLI, nessuno stato nascosto.
- **Pubblicazione one-shot su Hugging Face Hub**, model card generata dai metadati del run.

## Indice

- [Relazione con synThor](#relazione-con-synthor)
- [Struttura del progetto](#struttura-del-progetto)
- [Installazione](#installazione)
- [Quickstart](#quickstart)
- [LoRA vs full fine-tuning](#lora-vs-full-fine-tuning)
- [Configurazione di un run](#configurazione-di-un-run)
- [Output di un run](#output-di-un-run)
- [Metodologia di scoring](#metodologia-di-scoring)
- [Architettura](#architettura)
- [Pubblicazione su Hugging Face](#pubblicazione-su-hugging-face)
- [Limiti noti](#limiti-noti)
- [Test](#test)

## Relazione con synThor

I dati arrivano dal progetto gemello [synThor](../synThor), che genera documenti
sintetici (manifesti di carico, liste d'imbarco, documenti d'identità italiani) e li
esporta con:

```bash
python export_dataset.py -i output -o output/manifests.parquet \
    --format parquet --val-fraction 0.1
# oppure --format unsloth per dataset grandi, shardati sotto ~479MB/file
```

**trainmaster non importa codice da synThor**: l'unico accoppiamento è il contratto dello
schema Parquet (`id`, `language`, `document_type`, `image`, `instruction`,
`ground_truth`, `messages`). Se lo schema di synThor cambia in modo incompatibile, lo
scopre `src/trainmaster/data.py`, non un import rotto.

## Struttura del progetto

```
trainmaster/
  src/trainmaster/       # package (layout src/ standard)
    config.py, data.py, scoring.py, report.py, pipeline.py   # CPU-safe
    model.py, inference.py                                    # frontiera GPU (import locali)
    model_card.py                                              # model card per la pubblicazione HF
    cli/                                                        # entry point installati (vedi sotto)
  configs/                # RunConfig di esempio (YAML)
  tests/
  .github/workflows/      # CI (pytest, no GPU) — la pubblicazione HF è manuale, vedi sotto
```

Nessuno script alla radice: le CLI sono entry point installati (`[project.scripts]` in
`pyproject.toml`), lo standard di packaging Python quando il progetto è pip/uv-installabile
— elimina l'ambiguità "da dove lancio questo script".

## Installazione

Gestito con [uv](https://docs.astral.sh/uv/).

```bash
uv sync                    # core CPU-safe (config/data/scoring/report/pipeline) + pytest
uv sync --extra train      # aggiunge lo stack GPU: torch, unsloth, trl, transformers, ...
```

`torch`/`torchvision` dell'extra `train` arrivano dall'indice CUDA esplicito configurato
in `pyproject.toml` (`[tool.uv.sources]` + `[[tool.uv.index]]`, puntato a
`https://download.pytorch.org/whl/cu132` — build adatta a GPU Blackwell come le RTX 50);
il resto dell'extra viene da PyPI come di consueto. Cambiare indice per un'altra versione
CUDA si fa in un punto solo, in `pyproject.toml`.

L'intera test suite gira con il solo `uv sync` (nessuna GPU richiesta):

```bash
uv run pytest tests/ -v
```

## Quickstart

```bash
uv run trainmaster-train --config configs/default.yaml
```

Esegue: prepara i dati → addestra → salva il checkpoint → genera predizioni sulla
validazione → punteggia → scrive report. Override puntuali senza toccare lo YAML:

```bash
uv run trainmaster-train --config configs/default.yaml \
    --set training.learning_rate=1e-4 --set data.max_samples=200
```

Comandi separati, utili in iterazione:

```bash
# solo valutazione di un checkpoint già addestrato
uv run trainmaster-evaluate --config configs/default.yaml --checkpoint runs/default/checkpoint

# ripunteggia/riproduce il report da predizioni già generate, senza GPU
uv run trainmaster-score --predictions runs/default/eval/predictions.parquet -c configs/default.yaml
```

## LoRA vs full fine-tuning

Di default il training è LoRA (`lora.enabled: true`). Per il full fine-tuning (tutti i
pesi allenabili, nessun adapter):

```bash
uv run trainmaster-train --config configs/default.yaml \
    --set lora.enabled=false --set model.load_in_4bit=false
```

Internamente `lora.enabled=false` passa `full_finetuning=True` a
`FastLanguageModel.from_pretrained` (vedi `src/trainmaster/model.py`) invece di applicare
`get_peft_model`. Serve più VRAM del LoRA equivalente; su un modello da 0.8B resta comunque
alla portata di una singola GPU consumer.

## Configurazione di un run

`configs/default.yaml`:

```yaml
name: qwen3.5-manifest-extraction
data:
  train_files: "data/*_train*.parquet"        # glob: copre anche gli shard di --format unsloth
  validation_files: "data/*_validation*.parquet"
model:
  model_id: "Qwen/Qwen3.5-0.8B"
  load_in_4bit: false      # Unsloth sconsiglia QLoRA 4-bit su Qwen3.5
lora:
  enabled: true             # false = full fine-tuning
  r: 16
training:
  output_dir: "runs/default"
  num_train_epochs: 1.0
evaluation:
  enabled: true            # valutazione semantica JSON post-training
report:
  enabled: true            # report HTML predizioni vs ground truth
```

## Output di un run

```
runs/<name>/
  checkpoint/            # pesi (adapter LoRA o full) + processor salvati da model.py
  config.yaml            # RunConfig effettiva (con override applicati), per riproducibilità
  eval/
    predictions.parquet  # id/document_type/language/instruction/ground_truth/predicted_text/image
    metrics.json         # aggregate() per document_type e per language
    report.html          # immagine + ground truth + predizione affiancati, F1 crescente
```

`predictions.parquet` separa esplicitamente "genera predizioni" (serve GPU) da
"punteggia e produci report" (CPU puro): `trainmaster-score` ricalcola metriche/report da
lì senza mai ricaricare il modello.

## Metodologia di scoring

I 7 `document_type` di synThor si dividono in due famiglie (vedi
`src/trainmaster/scoring.py`):

- **`cargo_manifest`, `camion_list`, `veicols_list`, `passenger_list`** — confronto
  ricorsivo campo-per-campo (precision/recall/F1 sui leaf di dict/liste annidati),
  tolleranza numerica relativa configurabile (`evaluation.numeric_tolerance`).
- **`carta_identita`, `patente`, `tessera_sanitaria`** — match esatto sull'unico campo,
  `codice_fiscale`.

Un output del modello che non è JSON valido non fa crashare la valutazione: viene
segnalato con `parse_error=True` e punteggiato a zero.

## Architettura

| Modulo | Responsabilità | Import pesanti? |
|---|---|---|
| `config.py` | `RunConfig` (dataclass annidate frozen) + load/dump YAML + override CLI | no |
| `data.py` | Lettura Parquet (anche multi-shard) + ricostruzione conversazioni | no |
| `scoring.py` | Strategy + registry di scoring per document_type | no |
| `report.py` | Report HTML predizioni vs ground truth | no |
| `model_card.py` | Model card per la pubblicazione su Hugging Face | no |
| `model.py` | Wrapper Unsloth: carica modello/LoRA, costruisce l'SFTTrainer | **sì** (import locali) |
| `inference.py` | Loop di generazione sulla validazione | **sì** (import locali) |
| `pipeline.py` | Orchestrazione train→save→evaluate→report, con DI sui punti GPU | no |
| `cli/*.py` | Entry point installati, thin wrapper attorno ai moduli sopra | no |

**Solo `model.py` e `inference.py` importano `unsloth`/`trl`/`torch`, sempre dentro il
corpo delle funzioni.** È il seam che rende `pipeline.py` (e quindi l'intera logica di
orchestrazione, scoring e report) testabile con `pytest` senza GPU: `tests/test_pipeline.py`
inietta `model_loader`/`trainer_builder`/`inference_runner` fake e verifica che
`import trainmaster.pipeline` non trascini mai `torch` in `sys.modules`. Sfruttato anche
dalla CI (`.github/workflows/ci.yml`): la suite gira su un runner GitHub-hosted standard,
senza GPU.

## Pubblicazione su Hugging Face

Pubblicazione manuale da terminale, non da CI: niente runner self-hosted da installare
e mantenere, al prezzo di un token salvato in locale invece che a vita breve. La CLI
ufficiale `hf` è già disponibile nel venv del progetto (dipendenza transitiva di
`datasets`), non serve installarla a parte.

**Setup una tantum**:

1. Crea il repo modello su huggingface.co (es. `<tuo-username>/<nome-modello>`).
2. Crea un access token con scope **write**: huggingface.co → Settings → Access Tokens.
3. `uv run hf auth login` e incolla il token — resta salvato nella cache locale di
   `huggingface_hub` (`~/.cache/huggingface/token`), non nel repo.

**Dopo ogni training**:

```bash
uv run trainmaster-model-card --run-dir runs/default   # genera checkpoint/README.md
uv run hf upload <tuo-username>/<nome-modello> runs/default/checkpoint .
```

`trainmaster-model-card` legge `config.yaml` e (se presente) `eval/metrics.json` del run
e scrive la model card con iperparametri e metriche di validazione; `hf upload` carica
`checkpoint/` così com'è.

Restano su GitHub Actions (`.github/workflows/ci.yml`) solo i test, che non richiedono
GPU né accesso alla tua macchina — l'unica parte di CI/CD che vale la pena automatizzare
qui, dato che pubblicazione e training restano comandi locali.

## Limiti noti

- **Confronto liste posizionale per indice**: se il modello riordina righe valide di
  `cargo_items` o delle liste d'imbarco, lo score le penalizza come mancanti/aggiunte.
  Accettato per v1 (YAGNI); se le metriche sembrano ingiustamente basse rispetto al
  report visivo, è il primo sospetto.
- **Generazione non batchata** in `inference.py` (batch_size=1): sufficiente per la
  validazione periodica di un piccolo VLM, non ottimizzata per throughput.
- **API Unsloth per Qwen3.5 vision + `SFTTrainer`/`UnslothVisionDataCollator`**: la guida
  ufficiale Unsloth mostra `FastLanguageModel.from_pretrained`/`get_peft_model` per
  Qwen3.5, ma non il codice esatto del trainer per il ramo vision. `model.py` usa la
  combinazione documentata per il fine-tuning multimodale in generale — **da verificare
  con uno smoke test reale** (`configs/smoke_test.yaml`) prima di un run lungo; un issue
  GitHub aperto (`unslothai/unsloth#5845`) conferma che questo percorso è recente.

## Test

```bash
uv run pytest tests/ -v
```

Tutti i test girano senza GPU/torch/unsloth installati: `config.py`/`data.py`/
`scoring.py`/`report.py`/`model_card.py`/`pipeline.py` sono logica pura o
dependency-injected. `tests/conftest.py` costruisce dataset Parquet in-memory con lo
schema reale di synThor (incluso il caso multi-shard).
