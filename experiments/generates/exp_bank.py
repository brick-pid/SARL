from __future__ import annotations

import logging
import os
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class Experience:
    task: str
    summary: str

    @property
    def retrieval_text(self) -> str:
        return self.task


@dataclass
class TrajectoryExperience:
    task: str
    action_list: List[str]
    obs_list: List[str]

    @property
    def act_obs_traj(self) -> str:
        action_obs_pairs = "\n".join(
            f"Action: {a}\nObservation: {o}"
            for a, o in zip(self.action_list, self.obs_list)
        )
        return f"{self.task}\n{action_obs_pairs}"

    @property
    def recent_act_obs_traj(self) -> str:
        recent_turns = 10
        all_actions = "\n".join(f"Action: {action}" for action in self.action_list) or "Action: <empty>"
        recent_pairs = "\n".join(
            f"Action: {action}\nObservation: {obs}"
            for action, obs in zip(self.action_list[-recent_turns:], self.obs_list[-recent_turns:])
        )
        return f"{self.task}\n# All actions\n{all_actions}\n\n# Recent {recent_turns} turns\n{recent_pairs}"

    def update(self, action: str, obs: str):
        self.action_list.append(action)
        self.obs_list.append(obs)

class RemoteEmbeddingClient:
    DEFAULT_BASE_URL = "http://127.0.0.1:37001"
    DEFAULT_TIMEOUT = 180

    def __init__(self, base_url: str | None = None, timeout: int | None = None) -> None:
        self.base_url = (base_url or os.environ.get("EXPERIENCE_BANK_EMBEDDING_URL") or self.DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout or int(os.environ.get("EXPERIENCE_BANK_EMBEDDING_TIMEOUT", self.DEFAULT_TIMEOUT))
        import requests

        self._session = requests.Session()
        self._session.trust_env = False
        self._cache: dict[str, list[float]] = {}

    def embed(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        if not texts:
            return []

        if is_query:
            uncached_texts = [text for text in texts if text not in self._cache]
            texts_to_encode = uncached_texts
        else:
            texts_to_encode = texts

        if texts_to_encode:
            response = self._session.post(
                f"{self.base_url}/encode",
                json={"text": texts_to_encode},
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                raise ValueError(f"Unexpected embedding response payload: {type(payload)!r}")
            if len(payload) != len(texts_to_encode):
                raise ValueError(
                    f"Unexpected embedding response length: expected {len(texts_to_encode)}, got {len(payload)}"
                )
            if is_query:
                for text, item in zip(texts_to_encode, payload):
                    self._cache[text] = item["embedding"]
                return [self._cache[text] for text in texts]
            else:
                return [item["embedding"] for item in payload]

        return [self._cache[text] for text in texts]


class ExperienceBank:
    COLLECTION_NAME = "experiences"

    def __init__(self, dir: str, resume_experience_bank_path: str | None = None) -> None:
        self.dir = Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.storage_path = self.dir / "experience_bank.pkl"
        self._db_path = self.dir / "chroma"
        self._db_path.mkdir(parents=True, exist_ok=True)
        self._embedding_client = RemoteEmbeddingClient()

        if resume_experience_bank_path is not None:
            source_path = Path(resume_experience_bank_path)
            if not source_path.is_file():
                raise FileNotFoundError(f"resume_experience_bank_path does not exist: {source_path}")
            shutil.copy2(source_path, self.storage_path)
            logger.info("Copied experience bank from %s to %s", source_path, self.storage_path)

        self.experiences: list[Experience] = []
        if self.storage_path.exists():
            with self.storage_path.open("rb") as f:
                self.experiences = pickle.load(f)

        import chromadb

        client = chromadb.PersistentClient(path=str(self._db_path))
        collection_names = {
            collection.name if hasattr(collection, "name") else collection
            for collection in client.list_collections()
        }
        if self.COLLECTION_NAME in collection_names:
            client.delete_collection(self.COLLECTION_NAME)

        self._collection = client.create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        if self.experiences:
            documents = [exp.retrieval_text for exp in self.experiences]
            self._collection.add(
                ids=[f"exp_{idx}" for idx in range(len(self.experiences))],
                documents=documents,
                embeddings=self._embedding_client.embed(documents),
            )

    def add_experiences(self, new_experiences) -> None:
        if not new_experiences:
            return

        start_idx = len(self.experiences)
        self.experiences.extend(new_experiences)
        documents = [exp.retrieval_text for exp in new_experiences]
        self._collection.add(
            ids=[f"exp_{idx}" for idx in range(start_idx, len(self.experiences))],
            documents=documents,
            embeddings=self._embedding_client.embed(documents),
        )
        self.save()
        # breakpoint()

    def retrieve(self, query: str, top_k: int = 3, return_str: bool = True) -> str | list[Experience]:
        if top_k <= 0 or not self.experiences:
            return "" if return_str else []

        try:
            results = self._collection.query(
                query_embeddings=self._embedding_client.embed([query], is_query=True),
                n_results=min(top_k, len(self.experiences)),
            )
        except Exception:
            logger.exception("Experience retrieval failed for query: %s", query)
            return "" if return_str else []
        exps = []
        for exp_id in results.get("ids", [[]])[0]:
            assert exp_id.startswith("exp_"), "Unexpected experience ID format: {exp_id}"
            try:
                idx = int(exp_id.removeprefix("exp_"))
            except ValueError:
                continue
            if 0 <= idx < len(self.experiences):
                exps.append(self.experiences[idx])

        if return_str:
            return "\n\n".join(exp.summary for exp in exps)
        return exps

    def save(self) -> None:
        with self.storage_path.open("wb") as f:
            pickle.dump(self.experiences, f)
