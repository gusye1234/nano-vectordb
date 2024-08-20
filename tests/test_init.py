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
    r = a.query(np.random.rand(fake_dim), 10)
    print("Query", time() - start)
    assert len(r) > 0
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
