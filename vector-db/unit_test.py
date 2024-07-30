import numpy as np
import unittest
from simple_vector_db import VectorDatabase

class TestVectorDatabase(unittest.TestCase):

    def setUp(self):
        self.db = VectorDatabase()
        self.vectors = {
            "vector1": [1, 2, 3],
            "vector2": [4, 5, 6],
            "vector3": [7, 8, 9],
            "vector4": [1, 0, 0]
        }

        for id, vector in self.vectors.items():
            self.db.add_vector(id, vector)

    def test_add_vector(self):
        new_vector_id = "vector5"
        new_vector = [0, 1, 0]
        self.db.add_vector(new_vector_id, new_vector)
        self.assertIn(new_vector_id, self.db.vectors)
        np.testing.assert_array_equal(self.db.vectors[new_vector_id], np.array(new_vector))

    def test_search(self):
        query_vector = [1, 2, 3]
        results = self.db.search(query_vector, top_k = 3)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0][0], "vector1")


    def test_vector_index(self):
        self.assertTrue("vector2" in self.db.vector_index["vector1"])
        similarity = self.db.vector_index["vector1"]["vector2"]
        expected_similarity = np.dot(self.vectors["vector1"], self.vectors["vector2"])/(
            np.linalg.norm(self.vectors["vector1"]) * np.linalg.norm(self.vectors["vector2"]))
        self.assertAlmostEqual(similarity, expected_similarity, places=5)

    def test_invalid_vector_retrieval(self):
        vector = self.db.get_vector("non_existent")
        self.assertIsNone(vector)

if __name__ == 'main':
    unittest.main()
    
