from __future__ import annotations

import importlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence


@dataclass
class Experience:
    task: str
    action_list: List[str]
    obs_list: List[str]
    reward: float

    @property
    def act_obs_traj(self) -> str:
        """
        task + (act + obs) * n
        """
        action_obs_pairs = "\n".join(
            f"Action: {a}\nObservation: {o}"
            for a, o in zip(self.action_list, self.obs_list)
        )
        return f"{self.task}\n{action_obs_pairs}"

class Qwen3EmbeddingFunction:
    MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
    BATCH_SIZE = 16

    _model = None
    _device = None

    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        self.model_name = model_name or self.MODEL_NAME
        self.device = device

    def __call__(self, input: list[str]) -> list[list[float]]:
        model = self._load_model(self.model_name, self.device)
        embeddings = model.encode(
            input,
            normalize_embeddings=True,
            batch_size=self.BATCH_SIZE,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    @classmethod
    def _load_model(cls, model_name: str, device: str | None):
        if cls._model is not None and cls._device == device:
            return cls._model

        torch = importlib.import_module("torch")
        sentence_transformers = importlib.import_module("sentence_transformers")
        SentenceTransformer = sentence_transformers.SentenceTransformer

        if device is None:
            assert torch.cuda.is_available(), (
                "Chroma retrieval backend requires a CUDA GPU for embeddings."
            )
            device = "cuda"

        model = SentenceTransformer(model_name, device=device)
        cls._model = model
        cls._device = device
        return model


class ExperienceBank:
    COLLECTION_NAME = "experiences"

    def __init__(self, dir: str) -> None:
        self.dir = Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.storage_path = self.dir / "experience_bank.pkl"
        self._db_path = self.dir / "chroma"
        self.experiences: list[Experience] = []
        self._collection = None
        self._experience_ids: list[str] = []
        self._id_to_experience: dict[str, Experience] = {}

    def add(self, experience: Experience | Sequence[Experience]) -> None:
        new_experiences = [experience] if isinstance(experience, Experience) else list(experience)
        if not new_experiences:
            return

        start_idx = len(self.experiences)
        self.experiences.extend(new_experiences)
        self._append_to_index(new_experiences, start_idx)

    def retrieve(self, query: str, top_k: int = 3, return_str: bool = True) -> str | list[Experience]:
        if top_k <= 0 or not self.experiences:
            return "" if return_str else []

        exps = self._search(query, top_k=top_k)
        if return_str:
            return "\n\n".join(exp.act_obs_traj for exp in exps)
        return exps

    def save(self) -> None:
        with self.storage_path.open("wb") as f:
            pickle.dump(self.experiences, f)

    def load(self) -> None:
        if not self.storage_path.exists():
            self.experiences = []
            self._rebuild_index()
            return

        with self.storage_path.open("rb") as f:
            self.experiences = pickle.load(f)
        self._rebuild_index()

    def _append_to_index(self, experiences: Sequence[Experience], start_idx: int) -> None:
        if not experiences:
            return

        self._ensure_collection()
        new_ids = [f"exp_{idx}" for idx in range(start_idx, start_idx + len(experiences))]
        self._experience_ids.extend(new_ids)
        self._id_to_experience.update(
            {exp_id: exp for exp_id, exp in zip(new_ids, experiences)}
        )
        self._collection.add(
            ids=new_ids,
            documents=[exp.act_obs_traj for exp in experiences],
        )

    def _rebuild_index(self) -> None:
        self._experience_ids = []
        self._id_to_experience = {}

        client = self._create_client()
        collection_names = {
            collection.name if hasattr(collection, "name") else collection
            for collection in client.list_collections()
        }
        if self.COLLECTION_NAME in collection_names:
            client.delete_collection(self.COLLECTION_NAME)

        self._collection = None
        self._ensure_collection()
        self._append_to_index(self.experiences, start_idx=0)

    def _ensure_collection(self) -> None:
        if self._collection is not None:
            return

        client = self._create_client()
        self._collection = client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=Qwen3EmbeddingFunction(),
            metadata={"hnsw:space": "cosine"},
        )

    def _search(self, query: str, top_k: int) -> list[Experience]:
        if top_k <= 0 or self._collection is None or not self.experiences:
            return []

        results = self._collection.query(
            query_texts=[query],
            n_results=min(top_k, len(self.experiences)),
        )
        ids = results.get("ids", [[]])
        return [
            self._id_to_experience[exp_id]
            for exp_id in ids[0]
            if exp_id in self._id_to_experience
        ]

    def _create_client(self):
        chromadb = importlib.import_module("chromadb")
        self._db_path.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(self._db_path))
