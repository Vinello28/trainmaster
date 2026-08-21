# trainmaster — fine-tuning e validazione automatizzati di piccoli Vision-LLM

Automatizza il fine-tuning (via [Unsloth](https://unsloth.ai/)) e la validazione di
piccoli Vision-LLM open source su un task di estrazione documentale immagine → JSON
strutturato. Modello target: [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B),
un VLM nativo unificato (early-fusion testo+visione in un'unica architettura).

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
  .github/workflows/      # CI (pytest) + pubblicazione HF
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

## Anatomia di una config (`configs/default.yaml`)

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

## Pubblicazione su Hugging Face (CI/CD, Trusted Publisher)

`.github/workflows/publish.yml` pubblica un checkpoint già addestrato in locale su un
repo Hugging Face, **senza salvare alcun token**: usa i
[Trusted Publishers](https://huggingface.co/docs/hub/trusted-publishers) di HF, uno
scambio OIDC fra GitHub Actions e l'endpoint `/oauth/token` di HF che restituisce un
token scoperto al singolo repo, valido un'ora. Il training resta un comando locale
(`trainmaster-train`); il workflow è solo pubblicazione, a trigger manuale.

**Setup una tantum**:

1. Crea il repo modello su huggingface.co (es. `<tuo-username>/<nome-modello>`).
2. Su quel repo: **Settings → Trusted Publishers → Add** → provider "GitHub Actions",
   claim `repository = <owner>/<repo-github>`, `workflow = publish.yml` (e
   opzionalmente `branch = main` se il dispatch parte sempre da lì).
3. Sul repo GitHub: **Settings → Secrets and variables → Actions → Variables → New
   repository variable** → nome `HF_REPO_ID`, valore il repo id scelto al punto 1
   (è una variabile, non un secret: non è sensibile).
4. Registra un **runner self-hosted** sulla macchina con la GPU: repo GitHub →
   **Settings → Actions → Runners → New self-hosted runner**, segui i comandi che
   GitHub mostra per il tuo OS. È necessario perché il job deve vedere
   `runs/<name>/checkpoint/` sullo stesso filesystem dove hai allenato — i runner
   ospitati da GitHub non hanno accesso al tuo disco locale né una GPU.
5. `uv sync --extra train` sulla stessa macchina, poi alleni come sempre in locale.
6. Dalla tab **Actions** del repo GitHub, lanci manualmente "Publish to Hugging Face"
   indicando `run_dir` (es. `runs/default`).

Il workflow genera prima la model card (`trainmaster-model-card`, da `config.yaml` +
`eval/metrics.json` del run) e poi carica `checkpoint/` con la CLI ufficiale `hf upload`.

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
