import os
import numpy as np
from nano_vectordb import NanoVectorDB


def test_init():
    from time import time

    data_len = 1000
    fake_dim = 1024

    start = time()
    a = NanoVectorDB(fake_dim, storage_file="test.json")
    print("Load", time() - start)

    fake_embeds = np.random.rand(data_len, fake_dim)
    fakes_data = [{"__vector__": fake_embeds[i]} for i in range(data_len)]
    start = time()
    r = a.upsert(fakes_data)
    print("Upsert", time() - start)
    a.save()

    a = NanoVectorDB(fake_dim, storage_file="test.json")

    start = time()
    r = a.query(np.random.rand(fake_dim), 10)
    print("Query", time() - start)
    assert len(r) > 0
    print(r)
