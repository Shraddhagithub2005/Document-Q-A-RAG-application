"""
A tiny deterministic 'embedding function' used only in tests.

It avoids downloading Chroma's default ONNX model over the network, which
would make tests slow and flaky in CI. It's not semantically meaningful —
just consistent enough that identical/similar text hashes close together,
which is all these tests need.
"""
import hashlib

import numpy as np
from chromadb import Documents, EmbeddingFunction, Embeddings


class FakeEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        vectors = []
        for text in input:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            # Expand the 32-byte digest into a 64-dim float vector, deterministically.
            vec = np.frombuffer(digest * 2, dtype=np.uint8).astype(np.float32)
            vec = (vec - vec.mean()) / (vec.std() + 1e-6)
            vectors.append(vec.tolist())
        return vectors
