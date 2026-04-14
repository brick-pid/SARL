from __future__ import annotations

import json
import logging
import random
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib import request

import chromadb

logger = logging.getLogger(__name__)

EMBEDDING_PORTS = (37001, 37002)
EMBEDDING_TIMEOUT_SECONDS = 30
EMBEDDING_MAX_RETRIES = 3
EMBEDDING_RETRY_BACKOFF_SECONDS = (15, 30, 60)
CRITIQUE_REWARD_DIFF_THRESHOLD = 0.1
MAX_RETRIEVED_CRITIQUES = 10
DEFAULT_RETRIEVAL_TOP_K_TASKS = 3


@dataclass
class Trajectory:
    task_desc: str
    turn: int
    reward: float
    steps: list[dict[str, str]]

    def to_text(self, *, header: str | None = None, view: str = "default") -> str:
        if view not in {"default", "verifier"}:
            raise ValueError(f"Unsupported trajectory view: {view}")
        parts: list[str] = []
        if header:
            parts.append(header)
        parts.append(f"task: {self.task_desc}")
        if view == "default":
            parts.append(f"reward: {self.reward}")
        parts.append("steps:")
        if not self.steps:
            parts.append("<empty>")
            return "\n".join(parts)

        last_index = len(self.steps) - 1
        for idx, step in enumerate(self.steps):
            action = step.get("action", "")
            observation = step.get("observation", "")
            if view == "verifier" and idx == last_index:
                observation = "<hidden>"
            if len(observation) > 800:
                observation = observation[:800] + "..."
            parts.append(f"<action>{action}</action><observation>{observation}</observation>")
        return "\n".join(parts)


@dataclass
class Critique:
    task: str
    text: str
    reward_prev: float
    reward_after: float
    reward_diff: float


@dataclass
class TaskExperienceEntry:
    task_id: str
    task_desc: str
    trajectory: Trajectory | None = None
    critiques: list[Critique] = field(default_factory=list)

    def has_retrievable_content(self) -> bool:
        return self.trajectory is not None or bool(self.critiques)


@dataclass
class RetrievedExperienceContext:
    trajectories: list[Trajectory]
    critiques: list[Critique]

    def to_text(self, *, include_trajectories: bool = True) -> str:
        if not self.trajectories and not self.critiques:
            return ""

        sections: list[str] = []
        if include_trajectories and self.trajectories:
            traj_lines = [
                trajectory.to_text(header=f"Trajectory {idx}") for idx, trajectory in enumerate(self.trajectories, start=1)
            ]
            sections.append("# Retrieved Successful Trajectories\n" + "\n\n".join(traj_lines))

        if self.critiques:
            critique_lines = []
            for idx, critique in enumerate(self.critiques, start=1):
                critique_lines.append(
                    "\n".join(
                        [
                            f"## Critique {idx}",
                            f"task: {critique.task}",
                            f"reward_prev: {critique.reward_prev}",
                            f"reward_after: {critique.reward_after}",
                            critique.text,
                        ]
                    )
                )
            sections.append("# Retrieved Critiques\n" + "\n\n".join(critique_lines))
        return "\n\n".join(sections).strip()


class RemoteEmbeddingClient:
    def __init__(
        self,
        *,
        ports: tuple[int, ...] = EMBEDDING_PORTS,
        timeout: int = EMBEDDING_TIMEOUT_SECONDS,
        max_retries: int = EMBEDDING_MAX_RETRIES,
    ) -> None:
        self._ports = ports
        self._timeout = timeout
        self._max_retries = max_retries

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            port = random.choice(self._ports)
            try:
                return self._post_encode(texts, port)
            except Exception as exc:  # pragma: no cover - runtime/network path
                last_error = exc
                backoff_seconds = EMBEDDING_RETRY_BACKOFF_SECONDS[min(attempt - 1, len(EMBEDDING_RETRY_BACKOFF_SECONDS) - 1)]
                logger.warning(
                    "Embedding request failed on port %s (attempt %s/%s): %s",
                    port,
                    attempt,
                    self._max_retries,
                    exc,
                )
                if attempt < self._max_retries:
                    logger.warning("Retrying embedding request in %ss", backoff_seconds)
                    time.sleep(backoff_seconds)
        assert last_error is not None
        raise last_error

    def _post_encode(self, texts: list[str], port: int) -> list[list[float]]:
        payload = json.dumps({"text": texts}).encode("utf-8")
        req = request.Request(
            url=f"http://127.0.0.1:{port}/encode",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self._timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        if not isinstance(body, list) or len(body) != len(texts):
            raise ValueError(f"Unexpected embedding response for {len(texts)} texts")

        embeddings: list[list[float]] = []
        for item in body:
            if not isinstance(item, dict) or "embedding" not in item:
                raise ValueError("Embedding response item missing 'embedding'")
            embedding = item["embedding"]
            if not isinstance(embedding, list):
                raise ValueError("Embedding must be a list")
            embeddings.append([float(v) for v in embedding])
        return embeddings


class ExperienceBank:
    COLLECTION_NAME = "experience_bank_tasks"

    def __init__(
        self,
        bank_dir: str | Path,
        *,
        resume_experience_bank_path: str | None = None,
        embedding_client: RemoteEmbeddingClient | None = None,
    ) -> None:
        self.dir = Path(bank_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.storage_path = self.dir / "experience_bank.json"
        self.chroma_path = self.dir / "experience_bank_chroma"
        self._embedding_client = embedding_client or RemoteEmbeddingClient()
        self._entries: dict[str, TaskExperienceEntry] = {}

        if resume_experience_bank_path is not None and not self.storage_path.exists():
            source_path = Path(resume_experience_bank_path)
            if not source_path.is_file():
                raise FileNotFoundError(f"resume_experience_bank_path does not exist: {source_path}")
            shutil.copy2(source_path, self.storage_path)
            logger.info("Copied experience bank from %s to %s", source_path, self.storage_path)

        self._load()
        self._collection = self._init_collection()

    @property
    def entries(self) -> dict[str, TaskExperienceEntry]:
        return self._entries

    def retrieve(
        self,
        task_desc: str,
        *,
        top_k_tasks: int = DEFAULT_RETRIEVAL_TOP_K_TASKS,
        max_critiques: int = MAX_RETRIEVED_CRITIQUES,
    ) -> RetrievedExperienceContext:
        if top_k_tasks <= 0 or not task_desc or not self._entries:
            return RetrievedExperienceContext(trajectories=[], critiques=[])

        self._ensure_index_for_entries()
        collection_size = self._collection.count()
        if collection_size <= 0:
            return RetrievedExperienceContext(trajectories=[], critiques=[])

        query_embedding = self._embedding_client.embed([task_desc])[0]
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k_tasks, collection_size),
        )
        task_ids = result.get("ids", [[]])[0]
        selected_entries = [
            self._entries[task_id]
            for task_id in task_ids
            if task_id in self._entries and self._entries[task_id].has_retrievable_content()
        ]
        if not selected_entries:
            return RetrievedExperienceContext(trajectories=[], critiques=[])

        trajectories = [entry.trajectory for entry in selected_entries if entry.trajectory is not None]
        critiques: list[Critique] = []
        for entry in selected_entries:
            critiques.extend(entry.critiques)
        critiques.sort(key=lambda item: (item.reward_after == 1.0, item.reward_diff), reverse=True)
        return RetrievedExperienceContext(
            trajectories=trajectories,
            critiques=critiques[:max_critiques],
        )

    def update_success_trajectory(self, task_id: str, task_desc: str, trajectory: Trajectory | None) -> None:
        if trajectory is None or trajectory.reward != 1.0:
            return
        entry = self._get_or_create_entry(task_id, task_desc)
        current = entry.trajectory
        if current is None or trajectory.turn < current.turn:
            entry.trajectory = trajectory

    def add_critique(self, task_id: str, task_desc: str, critique: Critique | None) -> None:
        if critique is None or not critique.text.strip() or critique.reward_diff <= CRITIQUE_REWARD_DIFF_THRESHOLD:
            return
        entry = self._get_or_create_entry(task_id, task_desc)
        entry.critiques.append(critique)

    def save(self) -> None:
        payload = {
            "entries": {task_id: self._entry_to_dict(entry) for task_id, entry in self._entries.items()},
        }
        with self.storage_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        with self.storage_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        raw_entries = payload.get("entries", {})
        self._entries = {
            task_id: self._entry_from_dict(task_id, raw_entry)
            for task_id, raw_entry in raw_entries.items()
        }

    def _get_or_create_entry(self, task_id: str, task_desc: str) -> TaskExperienceEntry:
        entry = self._entries.get(task_id)
        if entry is None:
            entry = TaskExperienceEntry(task_id=task_id, task_desc=task_desc)
            self._entries[task_id] = entry
        return entry

    def _init_collection(self):
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self.chroma_path))
        collection_names = {
            collection.name if hasattr(collection, "name") else collection
            for collection in client.list_collections()
        }
        if self.COLLECTION_NAME in collection_names:
            return client.get_collection(self.COLLECTION_NAME)
        return client.create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def _ensure_index_for_entries(self) -> None:
        missing_ids = self._find_missing_collection_ids()
        if not missing_ids:
            return

        task_descs = [self._entries[task_id].task_desc for task_id in missing_ids]
        embeddings = self._embedding_client.embed(task_descs)
        self._collection.upsert(
            ids=missing_ids,
            documents=task_descs,
            embeddings=embeddings,
            metadatas=[{"task_id": task_id} for task_id in missing_ids],
        )

    def _find_missing_collection_ids(self) -> list[str]:
        entry_ids = list(self._entries)
        if not entry_ids:
            return []
        existing = self._collection.get(ids=entry_ids, include=[])
        existing_ids = set(existing.get("ids", []))
        return [task_id for task_id in entry_ids if task_id not in existing_ids]

    @staticmethod
    def _entry_to_dict(entry: TaskExperienceEntry) -> dict:
        return {
            "task_id": entry.task_id,
            "task_desc": entry.task_desc,
            "trajectory": asdict(entry.trajectory) if entry.trajectory is not None else None,
            "critiques": [asdict(critique) for critique in entry.critiques],
        }

    @staticmethod
    def _entry_from_dict(task_id: str, payload: dict) -> TaskExperienceEntry:
        raw_trajectory = payload.get("trajectory")
        trajectory = Trajectory(**raw_trajectory) if raw_trajectory is not None else None
        critiques = [
            Critique(
                task=str(item["task"]),
                text=str(item["text"]),
                reward_prev=float(item["reward_prev"]),
                reward_after=float(item["reward_after"]),
                reward_diff=float(item["reward_diff"]),
            )
            for item in payload.get("critiques", [])
        ]
        return TaskExperienceEntry(
            task_id=payload.get("task_id", task_id),
            task_desc=payload.get("task_desc", task_id),
            trajectory=trajectory,
            critiques=critiques,
        )


def load_experience_bank(config: dict) -> ExperienceBank:
    return ExperienceBank(
        config["exp_dir"],
        resume_experience_bank_path=config.get("resume_experience_bank_path"),
    )
