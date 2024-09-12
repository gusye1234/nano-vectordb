import os
import numpy as np
from nano_vectordb import NanoVectorDB


def test_init():
    from time import time

    data_len = 1000
    fake_dim = 1024

    start = time()
    a = NanoVectorDB(fake_dim)
    print("Load", time() - start)

    fake_embeds = np.random.rand(data_len, fake_dim)
    fakes_data = [{"__vector__": fake_embeds[i]} for i in range(data_len)]
    start = time()
    r = a.upsert(fakes_data)
    print("Upsert", time() - start)
    a.save()

    a = NanoVectorDB(fake_dim)

    start = time()
    r = a.query(np.random.rand(fake_dim), 10, better_than_threshold=0.01)
    print("Query", time() - start)
    print(r)
    os.remove("nano-vectordb.json")


def test_same_upsert():
    from time import time

    data_len = 1000
    fake_dim = 1024

    start = time()
    a = NanoVectorDB(fake_dim)
    print("Load", time() - start)

    fake_embeds = np.random.rand(data_len, fake_dim)
    fakes_data = [{"__vector__": fake_embeds[i]} for i in range(data_len)]
    r1 = a.upsert(fakes_data)
    assert len(r1["insert"]) == len(fakes_data)
    fakes_data = [{"__vector__": fake_embeds[i]} for i in range(data_len)]
    r2 = a.upsert(fakes_data)
    assert r2["update"] == r1["insert"]


def test_get():
    a = NanoVectorDB(1024)
    a.upsert(
        [
            {"__vector__": np.random.rand(1024), "__id__": str(i), "content": i}
            for i in range(100)
        ]
    )
    r = a.get(["0", "1", "2"])
    assert len(r) == 3
    assert r[0]["content"] == 0
    assert r[1]["content"] == 1
    assert r[2]["content"] == 2


def test_delete():
    a = NanoVectorDB(1024)
    a.upsert(
        [
            {"__vector__": np.random.rand(1024), "__id__": str(i), "content": i}
            for i in range(100)
        ]
    )
    a.delete(["0", "50", "99"])

    r = a.get(["0", "50", "99"])
    assert len(r) == 0
    assert len(a) == 97
