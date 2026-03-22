from dataclasses import asdict, dataclass
import hashlib
import json
import logging
from pathlib import Path
import re


LOGGER = logging.getLogger(__name__)


@dataclass
class SuccessRecipe:
    task_type: str
    prompt_summary: str
    language: str
    api_sequence: list[dict]
    total_calls: int
    total_errors: int
    score: float
    source_file: str


class MemoryPalace:
    _INDEX_VERSION = 1

    def __init__(self, runs_dir: Path, index_path: Path | None = None):
        self.runs_dir = Path(runs_dir)
        self.index_path = (
            Path(index_path) if index_path else self.runs_dir / "memory_index.json"
        )
        self._recipes_by_key: dict[str, SuccessRecipe] = {}

        if self.index_path.exists():
            if not self._load_index():
                self.index_traces()
        else:
            self.index_traces()

    def index_traces(self) -> int:
        self._recipes_by_key = {}
        indexed = 0

        if not self.runs_dir.exists():
            self._save_index()
            return 0

        for trace_path in sorted(self.runs_dir.glob("*.jsonl")):
            if trace_path.name == self.index_path.name:
                continue
            recipe = self._build_recipe_from_trace(trace_path)
            if not recipe:
                continue
            indexed += 1
            self._upsert_recipe(recipe)

        self._save_index()
        return indexed

    def find_recipe(self, task_type: str, prompt: str) -> SuccessRecipe | None:
        if not self._recipes_by_key:
            return None

        candidates = [
            r for r in self._recipes_by_key.values() if r.task_type == task_type
        ]
        if not candidates:
            return None

        desired_language = self._detect_language(prompt)
        same_language = [r for r in candidates if r.language == desired_language]
        pool = same_language if same_language else candidates

        return sorted(pool, key=self._recipe_sort_key)[0]

    def format_as_example(self, recipe: SuccessRecipe) -> str:
        lines: list[str] = []
        lines.append(
            f"## Proven recipe for {recipe.task_type} "
            f"({recipe.total_calls} calls, {recipe.total_errors} errors, score {recipe.score:.2f}, lang {recipe.language}):"
        )

        max_steps = 8
        for i, call in enumerate(recipe.api_sequence[:max_steps], start=1):
            method = str(call.get("method", "?")).upper()
            path = str(call.get("path", ""))
            body_shape = call.get("body_shape", {})
            body_compact = self._compact_shape(body_shape)
            if body_compact and body_compact != "{}":
                lines.append(f"{i}. {method} {path} {body_compact}")
            else:
                lines.append(f"{i}. {method} {path}")

        if len(recipe.api_sequence) > max_steps:
            lines.append(f"... +{len(recipe.api_sequence) - max_steps} more calls")

        lines.append(
            "Result: Perfect score run. Reuse call order/payload SHAPE only; replace entity values for current task."
        )
        return "\n".join(lines)

    def add_trace(self, trace_path: Path) -> bool:
        recipe = self._build_recipe_from_trace(Path(trace_path))
        if not recipe:
            return False
        self._upsert_recipe(recipe)
        self._save_index()
        return True

    def _load_index(self) -> bool:
        try:
            raw = json.loads(self.index_path.read_text(encoding="utf-8"))
            recipes_raw = raw.get("recipes", []) if isinstance(raw, dict) else []
            loaded: dict[str, SuccessRecipe] = {}
            for item in recipes_raw:
                if not isinstance(item, dict):
                    continue
                recipe = SuccessRecipe(
                    task_type=str(item.get("task_type", "")),
                    prompt_summary=str(item.get("prompt_summary", "")),
                    language=str(item.get("language", "unknown")),
                    api_sequence=item.get("api_sequence", [])
                    if isinstance(item.get("api_sequence"), list)
                    else [],
                    total_calls=int(item.get("total_calls", 0)),
                    total_errors=int(item.get("total_errors", 0)),
                    score=float(item.get("score", 0.0)),
                    source_file=str(item.get("source_file", "")),
                )
                if not recipe.task_type:
                    continue
                loaded[self._key_for(recipe.task_type, recipe.language)] = recipe
            self._recipes_by_key = loaded
            return True
        except Exception as exc:
            LOGGER.warning("Failed loading memory index %s: %s", self.index_path, exc)
            self._recipes_by_key = {}
            return False

    def _save_index(self) -> None:
        payload = {
            "version": self._INDEX_VERSION,
            "recipes": [
                asdict(recipe)
                for recipe in sorted(
                    self._recipes_by_key.values(),
                    key=lambda r: (r.task_type, r.language),
                )
            ],
        }
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _upsert_recipe(self, recipe: SuccessRecipe) -> None:
        key = self._key_for(recipe.task_type, recipe.language)
        existing = self._recipes_by_key.get(key)
        if existing is None or self._is_better(recipe, existing):
            self._recipes_by_key[key] = recipe

    def _is_better(self, cand: SuccessRecipe, curr: SuccessRecipe) -> bool:
        cand_key = self._recipe_sort_key(cand)
        curr_key = self._recipe_sort_key(curr)
        if cand_key != curr_key:
            return cand_key < curr_key
        return self._stable_id(cand.source_file) < self._stable_id(curr.source_file)

    @staticmethod
    def _recipe_sort_key(recipe: SuccessRecipe) -> tuple:
        return (recipe.total_calls, recipe.total_errors, -recipe.score)

    @staticmethod
    def _key_for(task_type: str, language: str) -> str:
        return f"{task_type}::{language}"

    @staticmethod
    def _stable_id(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def _build_recipe_from_trace(self, trace_path: Path) -> SuccessRecipe | None:
        events = self._read_events(trace_path)
        if not events:
            return None

        init_prompt = ""
        task_type = ""
        total_calls = 0
        total_errors = 0
        api_sequence: list[dict] = []
        score = 0.0
        perfect = False

        for event in events:
            event_type = event.get("event_type")
            payload_obj = event.get("payload")
            payload = payload_obj if isinstance(payload_obj, dict) else {}

            if event_type == "init":
                init_prompt = str(payload.get("prompt", ""))

            elif event_type == "planner":
                task_type = str(payload.get("task_type", ""))

            elif event_type == "tool_start":
                arguments_obj = payload.get("arguments")
                arguments = arguments_obj if isinstance(arguments_obj, dict) else {}
                method = arguments.get("method")
                path = arguments.get("path")
                if not method or not path:
                    continue
                body_shape = self._shape_from_body(arguments.get("json_body"))
                api_sequence.append(
                    {
                        "method": str(method).upper(),
                        "path": self._sanitize_path(str(path)),
                        "body_shape": body_shape,
                    }
                )

            elif event_type == "done":
                total_calls = int(
                    payload.get("tripletex_call_count", payload.get("call_count", 0))
                    or 0
                )
                total_errors = int(
                    payload.get("tripletex_error_count", payload.get("error_count", 0))
                    or 0
                )

            elif event_type == "competition_scoring":
                score, perfect = self._extract_score_and_perfection(payload)

        if not task_type or not init_prompt or not perfect:
            return None

        if total_calls <= 0:
            total_calls = len(api_sequence)

        return SuccessRecipe(
            task_type=task_type,
            prompt_summary=self._summarize_prompt(init_prompt),
            language=self._detect_language(init_prompt),
            api_sequence=api_sequence,
            total_calls=total_calls,
            total_errors=total_errors,
            score=score,
            source_file=trace_path.name,
        )

    @staticmethod
    def _read_events(trace_path: Path) -> list[dict]:
        events: list[dict] = []
        try:
            for line in trace_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    events.append(obj)
        except Exception as exc:
            LOGGER.warning("Skipping unreadable trace %s: %s", trace_path, exc)
            return []
        return events

    @staticmethod
    def _extract_score_and_perfection(payload: dict) -> tuple[float, bool]:
        score_raw = payload.get("score_raw")
        score_max = payload.get("score_max")
        checks_passed = payload.get("checks_passed")
        checks_total = payload.get("checks_total")

        score_value = payload.get("normalized_score", score_raw)
        try:
            score = float(score_value if score_value is not None else 0.0)
        except Exception:
            score = 0.0

        conditions: list[bool] = []
        try:
            raw = float(str(score_raw))
            maxv = float(str(score_max))
            if maxv > 0:
                conditions.append(raw >= (maxv - 1e-9))
        except Exception:
            pass

        try:
            passed = int(str(checks_passed))
            total = int(str(checks_total))
            if total > 0:
                conditions.append(passed >= total)
        except Exception:
            pass

        return score, bool(conditions) and all(conditions)

    @staticmethod
    def _summarize_prompt(prompt: str) -> str:
        compact = re.sub(r"\s+", " ", prompt).strip()
        return compact[:200]

    @staticmethod
    def _sanitize_path(path: str) -> str:
        cleaned = re.sub(
            r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b",
            "{uuid}",
            path,
        )
        cleaned = re.sub(r"/(\d+)(?=/|$)", "/{id}", cleaned)
        cleaned = re.sub(r"=\d+(?=&|$)", "=<num>", cleaned)
        return cleaned

    def _shape_from_body(self, value):
        if value is None:
            return {}
        if isinstance(value, dict):
            out = {}
            for key in sorted(value.keys()):
                out[str(key)] = self._shape_from_body(value[key])
            return out
        if isinstance(value, list):
            if not value:
                return ["empty"]
            return [self._shape_from_body(value[0])]
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, int):
            return "int"
        if isinstance(value, float):
            return "float"
        if isinstance(value, str):
            return "str"
        return "unknown"

    def _compact_shape(self, shape) -> str:
        text = self._shape_to_text(shape)
        return text if len(text) <= 140 else f"{text[:137]}..."

    def _shape_to_text(self, shape) -> str:
        if isinstance(shape, dict):
            if not shape:
                return "{}"
            parts = []
            for key in sorted(shape.keys()):
                child = shape[key]
                if isinstance(child, (dict, list)):
                    parts.append(f"{key}:{self._shape_to_text(child)}")
                else:
                    parts.append(f"{key}:{child}")
            return "{" + ", ".join(parts) + "}"
        if isinstance(shape, list):
            if not shape:
                return "[]"
            return "[" + self._shape_to_text(shape[0]) + "]"
        return str(shape)

    @staticmethod
    def _detect_language(prompt: str) -> str:
        text = f" {prompt.lower()} "

        detectors = [
            (
                "es",
                [
                    r"\b(una|un|el|la|los|las|factura|cliente|cuenta|importe|iva|corrija|crear|proyecto)\b"
                ],
            ),
            (
                "fr",
                [
                    r"\b(le|la|les|facture|client|compte|montant|tva|corrigez|créer|projet)\b"
                ],
            ),
            (
                "de",
                [
                    r"\b(der|die|das|rechnung|kunde|konto|betrag|mwst|korrigieren|projekt|erstellen)\b"
                ],
            ),
            (
                "pt",
                [r"\b(uma|um|fatura|cliente|conta|valor|iva|corrigir|criar|projeto)\b"],
            ),
            (
                "nn",
                [
                    r"\b(ikkje|kva|korleis|månad|løn|rekneskap|opprett|tilsett|verksemd)\b"
                ],
            ),
            (
                "nb",
                [r"\b(ikke|hva|hvordan|måned|lønn|regnskap|opprett|ansatt|kunde)\b"],
            ),
            (
                "en",
                [
                    r"\b(the|and|invoice|customer|account|amount|vat|correct|create|project)\b"
                ],
            ),
        ]

        for lang, patterns in detectors:
            for pat in patterns:
                if re.search(pat, text):
                    return lang
        return "unknown"
