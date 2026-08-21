"""Frontiera Unsloth: unico punto del progetto (insieme a ``inference.py``) che conosce
``unsloth``/``trl``/``torch``.

Tutti gli import pesanti stanno **dentro il corpo delle funzioni**, mai a livello modulo:
è ciò che permette a ``pipeline.py`` di importare questo modulo (per usarne le funzioni
come default dei parametri di dependency injection) restando eseguibile senza GPU/CUDA
installati — requisito verificato da ``tests/test_pipeline.py``.

Rischio noto da riverificare in esecuzione (segnalato nel piano): le API vision di
Unsloth/trl (``FastVisionModel.get_peft_model``, ``UnslothVisionDataCollator``, i campi
di ``SFTConfig`` per il training vision) possono cambiare fra versioni. Il codice sotto
rispecchia il workflow documentato da Unsloth per il fine-tuning vision al momento della
scrittura; va riverificato contro la versione effettivamente installata (uno smoke test
manuale col notebook ufficiale Unsloth vision è il modo più rapido).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL.Image import Image

    from trainmaster.config import LoraConfig, ModelConfig, TrainingConfig

__all__ = ["UnslothModelHandle", "load_model", "build_trainer"]


@dataclass
class UnslothModelHandle:
    """Incapsula modello + processor Unsloth dietro un'interfaccia minima
    (``generate``/``save``) usata da ``inference.py`` e ``pipeline.py``."""

    model: Any
    processor: Any
    model_config: "ModelConfig"

    def generate(self, image: "Image", instruction: str, *, max_new_tokens: int) -> str:
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            image, prompt, add_special_tokens=False, return_tensors="pt"
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


def load_model(
    model_config: "ModelConfig",
    lora: "LoraConfig | None" = None,
    *,
    for_training: bool,
    checkpoint: Path | None = None,
    seed: int = 0,
) -> UnslothModelHandle:
    """Carica il modello base (o un checkpoint salvato, per la valutazione) e prepara
    LoRA quando ``for_training=True``.

    ``checkpoint`` sovrascrive ``model_config.model_id`` come sorgente: Unsloth rileva
    da solo un adapter LoRA salvato in quella directory (``adapter_config.json``).
    """
    from unsloth import FastVisionModel

    source = str(checkpoint) if checkpoint is not None else model_config.model_id
    model, processor = FastVisionModel.from_pretrained(
        source,
        load_in_4bit=model_config.load_in_4bit,
        use_gradient_checkpointing="unsloth" if for_training else False,
    )

    if for_training:
        if lora is None:
            raise ValueError("for_training=True richiede una LoraConfig")
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=model_config.finetune_vision_layers,
            finetune_language_layers=model_config.finetune_language_layers,
            finetune_attention_modules=model_config.finetune_attention_modules,
            finetune_mlp_modules=model_config.finetune_mlp_modules,
            r=lora.r,
            lora_alpha=lora.alpha,
            lora_dropout=lora.dropout,
            target_modules=list(lora.target_modules) if lora.target_modules else None,
            bias="none",
            random_state=seed,
        )
        FastVisionModel.for_training(model)
    else:
        FastVisionModel.for_inference(model)

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
    vedi ``data.build_conversations``) e le prepara lui stesso per il modello.
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
