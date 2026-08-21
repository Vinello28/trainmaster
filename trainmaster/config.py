"""Configurazione dei run: dataclass annidate frozen, caricate da YAML.

Stesso stile delle dataclass frozen di synThor (``ExportConfig``, ``GeneratorConfig``),
ma qui la superficie di iperparametri è ampia abbastanza da giustificare un file YAML
invece dei soli flag CLI. Il costruttore ricorsivo (``_build``) evita di scrivere un
parser per ciascuna sotto-dataclass: cammina ``dataclasses.fields`` risolvendo le
annotazioni con ``typing.get_type_hints`` (necessario perché con
``from __future__ import annotations`` le annotazioni sono stringhe, non tipi).
"""

from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

__all__ = [
    "DataConfig",
    "ModelConfig",
    "LoraConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "ReportConfig",
    "RunConfig",
    "load_config",
    "dump_config",
    "parse_cli_overrides",
]


@dataclass(frozen=True)
class DataConfig:
    #: Glob su uno o più file Parquet (supporta gli shard ``-00001-of-00003.parquet``
    #: prodotti da ``export_dataset.py --format unsloth``).
    train_files: str
    validation_files: str
    max_samples: int | None = None
    seed: int = 0


@dataclass(frozen=True)
class ModelConfig:
    # TODO: verificare che questo model_id sia disponibile per FastVisionModel.from_pretrained
    # nella versione di unsloth installata. "qwen3.5 0.8B" non è un repo id verificabile:
    # questo è il piccolo Qwen-VL più vicino tra quelli noti come supportati da Unsloth.
    model_id: str = "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit"
    load_in_4bit: bool = True
    max_seq_length: int = 2048
    finetune_vision_layers: bool = True
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True


@dataclass(frozen=True)
class LoraConfig:
    r: int = 16
    alpha: int = 16
    dropout: float = 0.0
    target_modules: tuple[str, ...] | None = None


@dataclass(frozen=True)
class TrainingConfig:
    output_dir: Path = Path("runs/default")
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    eval_strategy: str = "steps"
    eval_steps: int = 50
    save_steps: int = 50
    bf16: bool = True
    seed: int = 0


@dataclass(frozen=True)
class EvaluationConfig:
    enabled: bool = True
    max_new_tokens: int = 1024
    #: Tolleranza relativa per il confronto numerico in RecursiveFieldScorer.
    numeric_tolerance: float = 0.01
    #: Limite di campioni da valutare; None = l'intero split di validazione.
    sample_limit: int | None = None


@dataclass(frozen=True)
class ReportConfig:
    enabled: bool = True
    max_examples: int = 40


@dataclass(frozen=True)
class RunConfig:
    name: str
    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    report: ReportConfig = field(default_factory=ReportConfig)


def _is_dataclass_type(tp: Any) -> bool:
    return isinstance(tp, type) and dataclasses.is_dataclass(tp)


def _is_optional_union(hint: Any) -> tuple[bool, Any]:
    """Riconosce ``X | None`` / ``Optional[X]`` e restituisce (è_optional, X)."""
    origin = typing.get_origin(hint)
    if origin is typing.Union or origin is types.UnionType:
        args = [a for a in typing.get_args(hint) if a is not type(None)]
        if len(args) == 1:
            return True, args[0]
    return False, hint


def _convert(hint: Any, raw: Any) -> Any:
    is_optional, inner = _is_optional_union(hint)
    if is_optional:
        return None if raw is None else _convert(inner, raw)
    if _is_dataclass_type(inner):
        if not isinstance(raw, Mapping):
            raise ValueError(f"atteso un mapping per {inner.__name__}, trovato {raw!r}")
        return _build(inner, raw)
    if inner is Path:
        return Path(raw)
    origin = typing.get_origin(inner)
    if origin is tuple:
        args = typing.get_args(inner)
        elem_type = args[0] if args else str
        return tuple(_convert(elem_type, v) for v in raw)
    if inner is float and isinstance(raw, int):
        return float(raw)
    return raw


def _build(cls: type, data: Mapping[str, Any]) -> Any:
    hints = typing.get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        hint = hints[f.name]
        has_default = (
            f.default is not dataclasses.MISSING
            or f.default_factory is not dataclasses.MISSING  # type: ignore[misc]
        )
        if f.name in data:
            kwargs[f.name] = _convert(hint, data[f.name])
            continue
        _, inner = _is_optional_union(hint)
        if _is_dataclass_type(inner):
            # Sezione omessa: costruisce comunque la sotto-dataclass dai suoi default,
            # così un campo obbligatorio annidato (es. DataConfig.train_files) produce
            # comunque un errore chiaro invece di un KeyError più a valle.
            kwargs[f.name] = _build(inner, {})
        elif not has_default:
            raise ValueError(f"campo obbligatorio mancante: {cls.__qualname__}.{f.name}")
    return cls(**kwargs)


def _apply_overrides(raw: dict[str, Any], overrides: Mapping[str, str]) -> dict[str, Any]:
    for dotted_key, text in overrides.items():
        *path, leaf = dotted_key.split(".")
        node = raw
        for part in path:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(text)
    return raw


def parse_cli_overrides(pairs: list[str]) -> dict[str, str]:
    """Converte ``["training.learning_rate=1e-4", ...]`` in un dict per ``load_config``."""
    overrides: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"override non valido (atteso chiave.punto=valore): {pair!r}")
        key, _, value = pair.partition("=")
        overrides[key.strip()] = value.strip()
    return overrides


def load_config(path: Path, overrides: Mapping[str, str] | None = None) -> RunConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: atteso un mapping YAML alla radice")
    raw = _apply_overrides(raw, overrides or {})
    return _build(RunConfig, raw)


def _to_serializable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _to_serializable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_to_serializable(v) for v in value]
    if isinstance(value, Mapping):
        return {k: _to_serializable(v) for k, v in value.items()}
    return value


def dump_config(config: RunConfig, path: Path) -> None:
    """Scrive la config effettiva (comprese le sezioni omesse nello YAML sorgente e gli
    override applicati) in ``path``, per riproducibilità del run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(_to_serializable(config), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
