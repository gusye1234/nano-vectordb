import os
import pytest
import numpy as np
from nano_vectordb import NanoVectorDB, MultiTenantNanoVDB
from nano_vectordb.dbs import f_METRICS, f_ID, f_VECTOR


def test_init():
    from time import time

    data_len = 1000
    fake_dim = 1024

    start = time()
    a = NanoVectorDB(fake_dim)
    print("Load", time() - start)

    fake_embeds = np.random.rand(data_len, fake_dim)
    fakes_data = [{f_VECTOR: fake_embeds[i], f_ID: i} for i in range(data_len)]
    query_data = fake_embeds[data_len // 2]
    start = time()
    r = a.upsert(fakes_data)
    print("Upsert", time() - start)
    a.save()

    a = NanoVectorDB(fake_dim)

    start = time()
    r = a.query(query_data, 10, better_than_threshold=0.01)
    assert r[0][f_ID] == data_len // 2
    print(r)
    assert len(r) <= 10
    for d in r:
        assert d[f_METRICS] >= 0.01
    os.remove("nano-vectordb.json")


def test_same_upsert():
    from time import time

    data_len = 1000
    fake_dim = 1024

    start = time()
    a = NanoVectorDB(fake_dim)
    print("Load", time() - start)

    fake_embeds = np.random.rand(data_len, fake_dim)
    fakes_data = [{f_VECTOR: fake_embeds[i]} for i in range(data_len)]
    r1 = a.upsert(fakes_data)
    assert len(r1["insert"]) == len(fakes_data)
    fakes_data = [{f_VECTOR: fake_embeds[i]} for i in range(data_len)]
    r2 = a.upsert(fakes_data)
    assert r2["update"] == r1["insert"]


def test_get():
    a = NanoVectorDB(1024)
    a.upsert(
        [
            {f_VECTOR: np.random.rand(1024), f_ID: str(i), "content": i}
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
            {f_VECTOR: np.random.rand(1024), f_ID: str(i), "content": i}
            for i in range(100)
        ]
    )
    a.delete(["0", "50", "90"])

    r = a.get(["0", "50", "90"])
    assert len(r) == 0
    assert len(a) == 97


def test_cond_filter():
    data_len = 10
    fake_dim = 1024

    a = NanoVectorDB(fake_dim)
    fake_embeds = np.random.rand(data_len, fake_dim)
    cond_filer = lambda x: x[f_ID] == 1

    fakes_data = [{f_VECTOR: fake_embeds[i], f_ID: i} for i in range(data_len)]
    query_data = fake_embeds[data_len // 2]
    a.upsert(fakes_data)

    assert len(a) == data_len
    r = a.query(query_data, 10, better_than_threshold=0.01)
    assert r[0][f_ID] == data_len // 2

    r = a.query(query_data, 10, filter_lambda=cond_filer)
    assert r[0][f_ID] == 1


def test_additonal_data():
    data_len = 10
    fake_dim = 1024

    a = NanoVectorDB(fake_dim)

    a.store_additional_data(a=1, b=2, c=3)
    a.save()

    a = NanoVectorDB(fake_dim)
    assert a.get_additional_data() == {"a": 1, "b": 2, "c": 3}
    os.remove("nano-vectordb.json")


def remove_non_empty_dir(dir_path):
    for f in os.listdir(dir_path):
        os.remove(os.path.join(dir_path, f))
    os.rmdir(dir_path)


def test_multi_tenant():
    with pytest.raises(ValueError):
        multi_tenant = MultiTenantNanoVDB(1024, max_capacity=0)

    multi_tenant = MultiTenantNanoVDB(1024)
    tenant_id = multi_tenant.create_tenant()
    tenant = multi_tenant.get_tenant(tenant_id)

    tenant.store_additional_data(a=1, b=2, c=3)
    multi_tenant.save()

    multi_tenant = MultiTenantNanoVDB(1024)
    assert multi_tenant.contain_tenant(tenant_id)
    tenant = multi_tenant.get_tenant(tenant_id)
    assert tenant.get_additional_data() == {"a": 1, "b": 2, "c": 3}

    with pytest.raises(ValueError):
        multi_tenant.get_tenant("1")  # not a uuid

    multi_tenant = MultiTenantNanoVDB(1024, max_capacity=1)
    multi_tenant.create_tenant()
    multi_tenant.get_tenant(tenant_id)

    multi_tenant.delete_tenant(tenant_id)

    multi_tenant = MultiTenantNanoVDB(1024)
    assert not multi_tenant.contain_tenant(tenant_id)
    remove_non_empty_dir("nano_multi_tenant_storage")

    multi_tenant = MultiTenantNanoVDB(1024, max_capacity=1)
    multi_tenant.create_tenant()
    assert not os.path.exists("nano_multi_tenant_storage")
    multi_tenant.create_tenant()
    assert os.path.exists("nano_multi_tenant_storage")
    remove_non_empty_dir("nano_multi_tenant_storage")
