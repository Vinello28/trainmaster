"""Frontiera Unsloth: unico punto del progetto (insieme a ``inference.py``) che conosce
``unsloth``/``trl``/``torch``.

Tutti gli import pesanti stanno **dentro il corpo delle funzioni**, mai a livello modulo:
è ciò che permette a ``pipeline.py`` di importare questo modulo (per usarne le funzioni
come default dei parametri di dependency injection) restando eseguibile senza GPU/CUDA
installati — requisito verificato da ``tests/test_pipeline.py``.

Modello target: ``Qwen/Qwen3.5-0.8B``, un VLM nativo *unificato* (early-fusion
testo+visione in un'unica architettura, non una famiglia separata "-VL" come
Qwen2.5-VL/Qwen3-VL). La guida ufficiale Unsloth per Qwen3.5
(``unsloth.ai/docs/models/qwen3.5/fine-tune``) carica questa famiglia con
``FastLanguageModel`` (non ``FastVisionModel``) e passa ``full_finetuning`` come
parametro diretto di ``from_pretrained`` — è così che si ottiene il full fine-tuning
invece del LoRA, senza doverlo dedurre dall'assenza di un adapter.

Rischio noto, non nascosto: la documentazione ufficiale mostra ``from_pretrained`` e
``get_peft_model`` per Qwen3.5, ma non il codice esatto del trainer/collator per il ramo
vision. ``build_trainer`` sotto usa ``SFTTrainer`` + ``UnslothVisionDataCollator``
(la combinazione documentata per il fine-tuning multimodale in generale, coerente col
formato conversazioni prodotto da ``data.build_conversations``), ma è il punto di
maggiore incertezza residua — un issue GitHub aperto (``unslothai/unsloth#5845``)
conferma che "Qwen3.5 vision + full_finetuning" è un percorso recente. Verificare con
uno smoke test reale (``configs/smoke_test.yaml``) prima di un run lungo.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL.Image import Image

    from trainmaster.config import LoraConfig, ModelConfig, TrainingConfig

__all__ = ["UnslothModelHandle", "load_model", "build_trainer"]

#: Target LoRA standard per Qwen3.5 (guida ufficiale Unsloth): nessuna distinzione
#: vision/language layer come nella vecchia API FastVisionModel — l'architettura
#: unificata non ha una vision tower separata da targetizzare a parte.
_DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass
class UnslothModelHandle:
    """Incapsula modello + processor Unsloth dietro un'interfaccia minima
    (``generate``/``save``) usata da ``inference.py`` e ``pipeline.py``."""

    model: Any
    processor: Any
    model_config: "ModelConfig"

    def generate(self, image: "Image", instruction: str, *, max_new_tokens: int) -> str:
        # L'immagine viaggia dentro i messaggi: apply_chat_template gestisce insieme
        # tokenizzazione e preprocessing immagine, indipendentemente dalla classe
        # esatta di processor restituita da FastLanguageModel per un modello
        # multimodale nativo (più robusto della chiamata separata
        # processor(image, prompt, ...) usata dalla vecchia API FastVisionModel).
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        input_length = inputs["input_ids"].shape[-1]
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True)
        generated_ids = output_ids[:, input_length:]
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))
        self.processor.save_pretrained(str(path))

    def save_merged(self, path: Path) -> None:
        """Fonde un adapter LoRA nei pesi base e salva un modello standalone in
        16-bit (nessuna quantizzazione 4-bit: sconsigliata da Unsloth per Qwen3.5,
        stessa scelta già fatta ovunque nel progetto, vedi ``ModelConfig.load_in_4bit``).
        ``save_pretrained_merged`` è un metodo che Unsloth applica dinamicamente
        all'istanza del modello dopo ``from_pretrained``/``get_peft_model``, non
        presente sulla classe ``FastLanguageModel``."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained_merged(str(path), tokenizer=self.processor, save_method="merged_16bit")


def load_model(
    model_config: "ModelConfig",
    lora: "LoraConfig | None" = None,
    *,
    for_training: bool,
    checkpoint: Path | None = None,
    seed: int = 0,
) -> UnslothModelHandle:
    """Carica il modello base (o un checkpoint salvato, per la valutazione).

    ``checkpoint`` sovrascrive ``model_config.model_id`` come sorgente: Unsloth rileva
    da solo un adapter LoRA salvato in quella directory (``adapter_config.json``).

    ``lora.enabled=False`` (con ``for_training=True``) richiede il full fine-tuning
    tramite ``full_finetuning=True`` su ``from_pretrained``: nessun adapter viene
    applicato, tutti i pesi restano allenabili.
    """
    from unsloth import FastLanguageModel

    source = str(checkpoint) if checkpoint is not None else model_config.model_id
    full_finetuning = for_training and lora is not None and not lora.enabled

    if for_training and model_config.load_in_4bit:
        warnings.warn(
            "Unsloth sconsiglia esplicitamente QLoRA 4-bit su Qwen3.5; "
            "considera model.load_in_4bit=false (il default del progetto)."
        )

    model, processor = FastLanguageModel.from_pretrained(
        source,
        max_seq_length=model_config.max_seq_length,
        load_in_4bit=model_config.load_in_4bit,
        load_in_16bit=not model_config.load_in_4bit,
        full_finetuning=full_finetuning,
    )

    if for_training and not full_finetuning:
        if lora is None:
            raise ValueError("for_training=True richiede una LoraConfig")
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora.r,
            lora_alpha=lora.alpha,
            lora_dropout=lora.dropout,
            target_modules=list(lora.target_modules) if lora.target_modules else list(_DEFAULT_TARGET_MODULES),
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=seed,
            max_seq_length=model_config.max_seq_length,
        )
    elif not for_training:
        FastLanguageModel.for_inference(model)

    return UnslothModelHandle(model=model, processor=processor, model_config=model_config)


def build_trainer(
    handle: UnslothModelHandle,
    train_conversations: list[dict[str, Any]],
    eval_conversations: list[dict[str, Any]] | None,
    training: "TrainingConfig",
) -> Any:
    """Costruisce l'``SFTTrainer`` di trl con il collator vision di Unsloth.

    ``dataset_kwargs={"skip_prepare_dataset": True}`` e ``remove_unused_columns=False``
    sono richiesti dal workflow vision documentato da Unsloth: il collator riceve le
    conversazioni grezze (lista di dict, non un ``datasets.Dataset`` con schema fisso,
    vedi ``data.build_conversations``) e le prepara lui stesso per il modello. Vedi la
    nota di modulo sull'incertezza residua di questa combinazione per Qwen3.5.
    """
    from trl import SFTConfig, SFTTrainer
    from unsloth.trainer import UnslothVisionDataCollator

    args = SFTConfig(
        output_dir=str(training.output_dir),
        per_device_train_batch_size=training.per_device_train_batch_size,
        gradient_accumulation_steps=training.gradient_accumulation_steps,
        num_train_epochs=training.num_train_epochs,
        learning_rate=training.learning_rate,
        warmup_ratio=training.warmup_ratio,
        eval_strategy=training.eval_strategy if eval_conversations else "no",
        eval_steps=training.eval_steps,
        save_strategy="steps",
        save_steps=training.save_steps,
        bf16=training.bf16,
        seed=training.seed,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=handle.model_config.max_seq_length,
        report_to=[],
    )
    return SFTTrainer(
        model=handle.model,
        tokenizer=handle.processor,
        data_collator=UnslothVisionDataCollator(handle.model, handle.processor),
        train_dataset=train_conversations,
        eval_dataset=eval_conversations,
        args=args,
    )
