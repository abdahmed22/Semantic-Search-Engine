from typing import Annotated
import numpy as np
import os

from product_quantization import ProductQuantization
from inverted_flat_index import InvertedFlatIndex

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70


class VectorDataBase:
    def __init__(self, d: int, m: int, pqk: int, ivfk: int, probes: int,
                 database_file_path="./assets/databases/saved_db_100k.csv",
                 new_db=True, db_size=None) -> None:
        self.D = d
        self.M = m
        self.PQK = pqk
        self.IVFK = ivfk
        self.Probes = probes
        self.db_path = database_file_path
        self.database_size = db_size

        self.pq = ProductQuantization(self.D, self.M, self.PQK)
        self.ivf = InvertedFlatIndex(self.D, self.IVFK, self.Probes)

        if new_db:
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.generate_database()

    def generate_database(self) -> None:
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((self.database_size, DIMENSION), dtype=np.float32)
        self._write_vectors_to_file(vectors)
        self._build_index()

    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='w+', shape=vectors.shape)
        mmap_vectors[:] = vectors[:]
        mmap_vectors.flush()

    def _get_num_records(self) -> int:
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    def insert_records(self, rows: Annotated[np.ndarray, (int, 70)]):
        num_old_records = self._get_num_records()
        num_new_records = len(rows)
        full_shape = (num_old_records + num_new_records, DIMENSION)
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=full_shape)
        mmap_vectors[num_old_records:] = rows
        mmap_vectors.flush()

        self._build_index()

    def get_one_row(self, row_num: int) -> np.ndarray:
        # This function is only load one row in memory
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            return f"An error occurred: {e}"

    def get_all_rows(self) -> np.ndarray:
        # Take care this load all the data in memory
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)

    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k=5):
        self.pq.load_centroids()
        self.pq.load_codebook()
        self.ivf.load_centroids()

        quantized_vectors = self.ivf.search(query, self.pq.M)
        #vectors = self.pq.search(quantized_vectors)

        #return vectors

    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def _build_index(self):
        database = self.get_all_rows()
        codebook = self.pq.generate_product_quantization(database)
        self.ivf.generate_inverted_flat_index(database, codebook)

