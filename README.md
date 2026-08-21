# trainmaster — fine-tuning e validazione automatizzati di piccoli Vision-LLM

Automatizza il fine-tuning (via [Unsloth](https://unsloth.ai/), `FastVisionModel` + LoRA)
e la validazione di piccoli Vision-LLM open source su un task di estrazione documentale
immagine → JSON strutturato.

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
scopre `trainmaster/data.py`, non un import rotto.

## Installazione

Gestito con [uv](https://docs.astral.sh/uv/).

```bash
uv sync                    # core CPU-safe (config/data/scoring/report/pipeline) + pytest
uv sync --extra train      # aggiunge lo stack GPU: torch, unsloth, trl, transformers, ...
```

Lo stack `train` richiede una GPU CUDA; installarlo solo quando serve davvero addestrare
o valutare. L'intera test suite gira con il solo `uv sync` (nessuna GPU richiesta):

```bash
uv run pytest tests/ -v
```

## Quickstart

```bash
uv run python train.py --config configs/default.yaml
```

Esegue: prepara i dati → addestra (LoRA) → salva il checkpoint → genera predizioni sulla
validazione → punteggia → scrive report. Override puntuali senza toccare lo YAML:

```bash
uv run python train.py --config configs/default.yaml \
    --set training.learning_rate=1e-4 --set data.max_samples=200
```

Comandi separati, utili in iterazione:

```bash
# solo valutazione di un checkpoint già addestrato
uv run python evaluate.py --config configs/default.yaml --checkpoint runs/default/checkpoint

# ripunteggia/riproduce il report da predizioni già generate, senza GPU
uv run python score.py --predictions runs/default/eval/predictions.parquet -c configs/default.yaml
```

## Anatomia di una config (`configs/default.yaml`)

```yaml
name: qwen-vl-manifest-extraction
data:
  train_files: "data/*_train*.parquet"        # glob: copre anche gli shard di --format unsloth
  validation_files: "data/*_validation*.parquet"
model:
  model_id: "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit"   # TODO: verificare/sostituire
lora:
  r: 16
training:
  output_dir: "runs/default"
  num_train_epochs: 1.0
evaluation:
  enabled: true            # valutazione semantica JSON post-training
report:
  enabled: true            # report HTML predizioni vs ground truth
```

Il `model_id` di default è un piccolo Qwen-VL noto come supportato da Unsloth: **non**
corrisponde a un id verificato per l'esatto modello richiesto originariamente
("qwen3.5 0.8B", non riscontrabile) — sostituirlo con l'id esatto voluto prima di un run
reale.

## Output di un run

```
runs/<name>/
  checkpoint/            # adapter LoRA + processor salvati da model.py
  config.yaml            # RunConfig effettiva (con override applicati), per riproducibilità
  eval/
    predictions.parquet  # id/document_type/language/instruction/ground_truth/predicted_text/image
    metrics.json         # aggregate() per document_type e per language
    report.html          # immagine + ground truth + predizione affiancati, F1 crescente
```

`predictions.parquet` separa esplicitamente "genera predizioni" (serve GPU) da
"punteggia e produci report" (CPU puro): `score.py` ricalcola metriche/report da lì senza
mai ricaricare il modello.

## Metodologia di scoring

I 7 `document_type` di synThor si dividono in due famiglie (vedi
`trainmaster/scoring.py`):

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
| `trainmaster/config.py` | `RunConfig` (dataclass annidate frozen) + load/dump YAML + override CLI | no |
| `trainmaster/data.py` | Lettura Parquet (anche multi-shard) + ricostruzione conversazioni | no |
| `trainmaster/scoring.py` | Strategy + registry di scoring per document_type | no |
| `trainmaster/report.py` | Report HTML predizioni vs ground truth | no |
| `trainmaster/model.py` | Wrapper Unsloth: carica modello/LoRA, costruisce l'SFTTrainer | **sì** (import locali) |
| `trainmaster/inference.py` | Loop di generazione sulla validazione | **sì** (import locali) |
| `trainmaster/pipeline.py` | Orchestrazione train→save→evaluate→report, con DI sui punti GPU | no |

**Solo `model.py` e `inference.py` importano `unsloth`/`trl`/`torch`, sempre dentro il
corpo delle funzioni.** È il seam che rende `pipeline.py` (e quindi l'intera logica di
orchestrazione, scoring e report) testabile con `pytest` senza GPU: `tests/test_pipeline.py`
inietta `model_loader`/`trainer_builder`/`inference_runner` fake e verifica che
`import trainmaster.pipeline` non trascini mai `torch` in `sys.modules`.

## Limiti noti

- **Confronto liste posizionale per indice**: se il modello riordina righe valide di
  `cargo_items` o delle liste d'imbarco, lo score le penalizza come mancanti/aggiunte.
  Accettato per v1 (YAGNI); se le metriche sembrano ingiustamente basse rispetto al
  report visivo, è il primo sospetto.
- **`model_id` di default da verificare**: vedi sopra.
- **Generazione non batchata** in `inference.py` (batch_size=1): sufficiente per la
  validazione periodica di un piccolo VLM, non ottimizzata per throughput.
- **API vision di Unsloth/trl**: `model.py` rispecchia il workflow vision documentato da
  Unsloth al momento della scrittura; le firme esatte (`get_peft_model`,
  `UnslothVisionDataCollator`, i campi di `SFTConfig`) vanno riverificate contro la
  versione effettivamente installata prima di un primo run reale.

## Test

```bash
uv run pytest tests/ -v
```

Tutti i test girano senza GPU/torch/unsloth installati: `config.py`/`data.py`/
`scoring.py`/`report.py`/`pipeline.py` sono logica pura o dependency-injected.
`tests/conftest.py` costruisce dataset Parquet in-memory con lo schema reale di synThor
(incluso il caso multi-shard).
