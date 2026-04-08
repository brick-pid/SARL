from __future__ import annotations

import logging
import os
import pickle
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_EXPERIENCE_BANK: "ExperienceBank | None" = None


@dataclass
class Experience:
    task: str
    action_list: list[str]
    obs_list: list[str]
    reward: float | None = None

    @property
    def retrieval_text(self) -> str:
        return self.task

    @property
    def act_obs_traj(self) -> str:
        res = f"task: {self.task}\n"
        res += f"reward: {self.reward}\n"
        res += "trajectory: "
        for a, o in zip(self.action_list, self.obs_list):
            if len(o) > 200:
                o = o[:200] + "..."
            res += f"-> <action>{a}</action> -> <observation>{o}</observation> "
        return res

    def update(self, action: str, obs: str) -> None:
        self.action_list.append(action)
        self.obs_list.append(obs)


class RemoteEmbeddingClient:
    DEFAULT_BASE_URL = "http://127.0.0.1:37001"
    DEFAULT_TIMEOUT = 30
    DEFAULT_MAX_RETRIES = 3

    def __init__(self, base_url: str | None = None, timeout: int | None = None, max_retries: int | None = None) -> None:
        self.base_url = (base_url or os.environ.get("EXPERIENCE_BANK_EMBEDDING_URL") or self.DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout or int(os.environ.get("EXPERIENCE_BANK_EMBEDDING_TIMEOUT", self.DEFAULT_TIMEOUT))
        self.max_retries = max_retries if max_retries is not None else self.DEFAULT_MAX_RETRIES
        self._cache: dict[str, list[float]] = {}

        import requests
        self._session = requests.Session()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        uncached_texts = [text for text in texts if text not in self._cache]
        if uncached_texts:
            payload = self._post_with_retry(uncached_texts)
            if not isinstance(payload, list):
                raise ValueError(f"Unexpected embedding response payload: {type(payload)!r}")
            if len(payload) != len(uncached_texts):
                raise ValueError(
                    f"Unexpected embedding response length: expected {len(uncached_texts)}, got {len(payload)}"
                )
            for text, item in zip(uncached_texts, payload):
                self._cache[text] = item["embedding"]
        return [self._cache[text] for text in texts]

    def _post_with_retry(self, texts: list[str]) -> list:
        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._session.post(
                    f"{self.base_url}/encode",
                    json={"text": texts},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                last_exc = e
                if attempt < self.max_retries:
                    wait = 2 ** attempt  # 2s, 4s
                    logger.warning("Embedding request failed (attempt %d/%d): %s, retrying in %ds", attempt, self.max_retries, e, wait)
                    time.sleep(wait)
                else:
                    logger.error("Embedding request failed after %d attempts: %s", self.max_retries, e)
        raise last_exc


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

    def add_experience(self, experience: Experience | None) -> None:
        if experience is None:
            return
        if experience.reward is None:
            raise ValueError("Experience reward must be set before adding to the bank")
        self.add_experiences([experience])

    def add_experiences(self, new_experiences: list[Experience]) -> None:
        if not new_experiences:
            return

        start_idx = len(self.experiences)
        self.experiences.extend(new_experiences)
        documents = [exp.retrieval_text for exp in new_experiences]
        try:
            embeddings = self._embedding_client.embed(documents)
            self._collection.add(
                ids=[f"exp_{idx}" for idx in range(start_idx, len(self.experiences))],
                documents=documents,
                embeddings=embeddings,
            )
        except Exception:
            logger.warning(
                "Embedding failed after retries, saving %d experience(s) to pickle only (skipping ChromaDB index)",
                len(new_experiences),
            )
        self.save()

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        return_str: bool = True,
    ) -> str | list[Experience]:
        if top_k <= 0 or not self.experiences:
            return "" if return_str else []

        try:
            results = self._collection.query(
                query_embeddings=self._embedding_client.embed([query]),
                n_results=min(20, len(self.experiences)),
            )
        except Exception:
            logger.exception("Experience retrieval failed for query: %s", query)
            return "" if return_str else []

        exps: list[Experience] = []
        for exp_id in results.get("ids", [[]])[0]:
            assert exp_id.startswith("exp_"), f"Unexpected experience ID format: {exp_id}"
            try:
                idx = int(exp_id.removeprefix("exp_"))
            except ValueError:
                continue
            if 0 <= idx < len(self.experiences):
                exps.append(self.experiences[idx])

        exps = sorted(exps, key=lambda exp: exp.reward if exp.reward is not None else float("-inf"), reverse=True)[:top_k]
        if return_str:
            return "\n\n".join(exp.act_obs_traj for exp in exps)
        return exps

    def save(self) -> None:
        with self.storage_path.open("wb") as f:
            pickle.dump(self.experiences, f)


def get_experience_bank(config: dict) -> ExperienceBank:
    global _EXPERIENCE_BANK

    if _EXPERIENCE_BANK is not None:
        return _EXPERIENCE_BANK

    bank = ExperienceBank(
        config["exp_dir"],
        resume_experience_bank_path=config.get("resume_experience_bank_path"),
    )
    _EXPERIENCE_BANK = bank
    return bank
