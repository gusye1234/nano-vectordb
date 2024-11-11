import os
import json
import base64
import hashlib
from uuid import uuid4
import numpy as np
from typing import TypedDict, Literal, Union, Callable
from dataclasses import dataclass, asdict
import sqlite3
import logging
from logging import getLogger

logging.basicConfig(level=logging.INFO)

f_ID = "__id__"
f_VECTOR = "__vector__"
f_METRICS = "__metrics__"
Data = TypedDict("Data", {"__id__": str, "__vector__": np.ndarray})
DataBase = TypedDict(
    "DataBase", {"embedding_dim": int, "data": list[Data], "matrix": np.ndarray}
)
Float = np.float32
ConditionLambda = Callable[[Data], bool]
logger = getLogger("nano-vectordb")


def array_to_buffer_string(array: np.ndarray) -> str:
    return base64.b64encode(array.tobytes()).decode()


def buffer_string_to_array(base64_str: str, dtype=Float) -> np.ndarray:
    return np.frombuffer(base64.b64decode(base64_str), dtype=dtype)


def load_storage(file_name) -> Union[DataBase, None]:
    if not os.path.exists(file_name):
        return None
    with open(file_name, encoding="utf-8") as f:
        data = json.load(f)
    data["matrix"] = buffer_string_to_array(data["matrix"]).reshape(
        -1, data["embedding_dim"]
    )
    logger.info(f"Load {data['matrix'].shape} data")
    return data


def hash_ndarray(a: np.ndarray) -> str:
    return hashlib.md5(a.tobytes()).hexdigest()


def normalize(a: np.ndarray) -> np.ndarray:
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


@dataclass
class NanoVectorDB:
    embedding_dim: int
    metric: Literal["cosine"] = "cosine"
    storage_file: str = "nano-vectordb.json"

    def pre_process(self):
        if self.metric == "cosine":
            self.__storage["matrix"] = normalize(self.__storage["matrix"])

    def __post_init__(self):
        default_storage = {
            "embedding_dim": self.embedding_dim,
            "data": [],
            "matrix": np.array([], dtype=Float).reshape(0, self.embedding_dim),
        }
        storage: DataBase = load_storage(self.storage_file) or default_storage
        assert (
            storage["embedding_dim"] == self.embedding_dim
        ), f"Embedding dim mismatch, expected: {self.embedding_dim}, but loaded: {storage['embedding_dim']}"
        self.__storage = storage
        self.usable_metrics = {
            "cosine": self._cosine_query,
        }
        assert self.metric in self.usable_metrics, f"Metric {self.metric} not supported"
        self.pre_process()
        logger.info(f"Init {asdict(self)} {len(self.__storage['data'])} data")

    def get_additional_data(self):
        return self.__storage.get("additional_data", {})

    def store_additional_data(self, **kwargs):
        self.__storage["additional_data"] = kwargs

    def upsert(self, datas: list[Data]):
        _index_datas = {
            data.get(f_ID, hash_ndarray(data[f_VECTOR])): data for data in datas
        }
        if self.metric == "cosine":
            for v in _index_datas.values():
                v[f_VECTOR] = normalize(v[f_VECTOR])
        report_return = {"update": [], "insert": []}
        for i, already_data in enumerate(self.__storage["data"]):
            if already_data[f_ID] in _index_datas:
                update_d = _index_datas.pop(already_data[f_ID])
                self.__storage["matrix"][i] = update_d[f_VECTOR].astype(Float)
                del update_d[f_VECTOR]
                self.__storage["data"][i] = update_d
                report_return["update"].append(already_data[f_ID])
        if len(_index_datas) == 0:
            return report_return
        report_return["insert"].extend(list(_index_datas.keys()))
        new_matrix = np.array(
            [data[f_VECTOR] for data in _index_datas.values()], dtype=Float
        )
        new_datas = []
        for new_k, new_d in _index_datas.items():
            del new_d[f_VECTOR]
            new_d[f_ID] = new_k
            new_datas.append(new_d)
        self.__storage["data"].extend(new_datas)
        self.__storage["matrix"] = np.vstack([self.__storage["matrix"], new_matrix])
        return report_return

    def get(self, ids: list[str]):
        return [data for data in self.__storage["data"] if data[f_ID] in ids]

    def delete(self, ids: list[str]):
        ids = set(ids)
        left_data = []
        delete_index = []
        for i, data in enumerate(self.__storage["data"]):
            if data[f_ID] in ids:
                delete_index.append(i)
                ids.remove(data[f_ID])
            else:
                left_data.append(data)
        self.__storage["data"] = left_data
        self.__storage["matrix"] = np.delete(
            self.__storage["matrix"], delete_index, axis=0
        )

    def save(self):
        storage = {
            **self.__storage,
            "matrix": array_to_buffer_string(self.__storage["matrix"]),
        }
        with open(self.storage_file, "w", encoding="utf-8") as f:
            json.dump(storage, f, ensure_ascii=False)

    def __len__(self):
        return len(self.__storage["data"])

    def query(
        self,
        query: np.ndarray,
        top_k: int = 10,
        better_than_threshold: float = None,
        filter_lambda: ConditionLambda = None,
    ) -> list[dict]:
        return self.usable_metrics[self.metric](
            query, top_k, better_than_threshold, filter_lambda=filter_lambda
        )

    def _cosine_query(
        self,
        query: np.ndarray,
        top_k: int,
        better_than_threshold: float,
        filter_lambda: ConditionLambda = None,
    ):
        query = normalize(query)
        if filter_lambda is None:
            use_matrix = self.__storage["matrix"]
            filter_index = np.arange(len(self.__storage["data"]))
        else:
            filter_index = np.array(
                [
                    i
                    for i, data in enumerate(self.__storage["data"])
                    if filter_lambda(data)
                ]
            )
            use_matrix = self.__storage["matrix"][filter_index]
        scores = np.dot(use_matrix, query)
        sort_index = np.argsort(scores)[-top_k:]
        sort_index = sort_index[::-1]
        sort_abs_index = filter_index[sort_index]
        results = []
        for abs_i, rel_i in zip(sort_abs_index, sort_index):
            if (
                better_than_threshold is not None
                and scores[rel_i] < better_than_threshold
            ):
                break
            results.append({**self.__storage["data"][abs_i], f_METRICS: scores[rel_i]})
        return results


@dataclass
class MultiTenantNanoVDB:
    embedding_dim: int
    metric: Literal["cosine"] = "cosine"
    max_capacity: int = 1000
    storage_dir: str = "./nano_multi_tenant_storage"

    @staticmethod
    def jsonfile_from_id(tenant_id):
        return f"nanovdb_{tenant_id}.json"

    def __post_init__(self):
        if self.max_capacity < 1:
            raise ValueError("max_capacity should be greater than 0")
        self.__storage: dict[str, NanoVectorDB] = {}
        self.__cache_queue: list[str] = []

    def contain_tenant(self, tenant_id: str) -> bool:
        return tenant_id in self.__storage or os.path.exists(
            f"{self.storage_dir}/{self.jsonfile_from_id(tenant_id)}"
        )

    def __load_tenant_in_cache(
        self, tenant_id: str, in_memory_tenant: NanoVectorDB
    ) -> NanoVectorDB:
        print(len(self.__storage), self.max_capacity)
        if len(self.__storage) >= self.max_capacity:
            vdb = self.__storage.pop(self.__cache_queue.pop(0))
            if not os.path.exists(self.storage_dir):
                os.makedirs(self.storage_dir)
            vdb.save()
        self.__storage[tenant_id] = in_memory_tenant
        self.__cache_queue.append(tenant_id)
        pass

    def __load_tenant(self, tenant_id: str) -> NanoVectorDB:
        if tenant_id in self.__storage:
            return self.__storage[tenant_id]
        if not self.contain_tenant(tenant_id):
            raise ValueError(f"Tenant {tenant_id} not in storage")

        in_memory_tenant = NanoVectorDB(
            self.embedding_dim,
            metric=self.metric,
            storage_file=f"{self.storage_dir}/{self.jsonfile_from_id(tenant_id)}",
        )
        self.__load_tenant_in_cache(tenant_id, in_memory_tenant)
        return in_memory_tenant

    def create_tenant(self) -> str:
        tenant_id = str(uuid4())
        in_memory_tenant = NanoVectorDB(
            self.embedding_dim,
            metric=self.metric,
            storage_file=f"{self.storage_dir}/{self.jsonfile_from_id(tenant_id)}",
        )
        self.__load_tenant_in_cache(tenant_id, in_memory_tenant)
        return tenant_id

    def delete_tenant(self, tenant_id: str):
        if tenant_id in self.__storage:
            self.__storage.pop(tenant_id)
            self.__cache_queue.remove(tenant_id)
        if os.path.exists(f"{self.storage_dir}/{self.jsonfile_from_id(tenant_id)}"):
            os.remove(f"{self.storage_dir}/{self.jsonfile_from_id(tenant_id)}")

    def get_tenant(self, tenant_id: str) -> NanoVectorDB:
        return self.__load_tenant(tenant_id)

    def save(self):
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)
        for db in self.__storage.values():
            db.save()
