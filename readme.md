<div align="center">
  <h1>nano-VectorDB</h1>
  <p><strong>A simple, easy-to-hack Vector Database</strong></p>
  <p>
    <img src="https://img.shields.io/badge/python->=3.9.11-blue">
    <a href="https://pypi.org/project/nano-vectordb/">
      <img src="https://img.shields.io/pypi/v/nano-vectordb.svg">
    </a>
    <a href="https://codecov.io/github/gusye1234/nano-vectordb" > 
 <img src="https://codecov.io/github/gusye1234/nano-vectordb/graph/badge.svg?token=3ACScwuv4h"/> 
 </a>
  </p>
</div>




🌬️ A vector database implementation with single-dependency (`numpy`).

🎁 It can handle a query from `100,000` vectors and return in 100 milliseconds.

🏃 It's okay for your prototypes, maybe even more.



## Install

**Install from PyPi**

```shell
pip install nano-vectordb
```

**Install from source**

```shell
# clone this repo first
cd nano-vectordb
pip install -e .
```



## Quick Start

**Faking your data**:

```python
from nano_vectordb import NanoVectorDB
import numpy as np

data_len = 100_000
fake_dim = 1024
fake_embeds = np.random.rand(data_len, fake_dim)    

fakes_data = [{"__vector__": fake_embeds[i], **ANYFIELDS} for i in range(data_len)]
```

You can add any fields to a data. But there are two keywords:

- `__id__`: If passed, `NanoVectorDB` will use your id, otherwise a generated id will be used.
- `__vector__`: must pass, your embedding `np.ndarray`.

**Init a DB**:

```python
vdb = NanoVectorDB(fake_dim, storage_file="fool.json")
```

Next time you init `vdb` from `fool.json`, `NanoVectorDB` will load the index automatically.

**Upsert**:

```python
r = vdb.upsert(fakes_data)
print(r["update"], r["insert"])
```

**Query**:

```python
print(vdb.query(np.random.rand(fake_dim)))
```

**Save**:

```python
# will create/overwrite 'fool.json'
vdb.save()
```

**Get, Delete**:

```python
# get and delete the inserted data
print(vdb.get(r["insert"]))
vdb.delete(r["insert"])
```



## Benchmark

> Embedding Dim: 1024. Device: MacBook M3 Pro

- Save a index with `100,000` vectors will generate a roughly 520M json file.
- Insert `100,000` vectors will cost roughly `2`s
- Query from `100,000` vectors will cost roughly `0.1`s
