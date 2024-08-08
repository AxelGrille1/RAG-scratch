import numpy as np

class VectorDatabase:
    def __init__(self):
        self.vectors = {}
        self.metadata = {}
        self.vector_index = {}  # Initialize the vector_index

    def add_vector(self, id, vector, metadata=None):
        self.vectors[id] = np.array(vector)  # Ensure vectors are numpy arrays
        if metadata:
            self.metadata[id] = metadata
        self._update_index(id, vector)  # Update index after adding a new vector

    def get_vector(self, id):
        return self.vectors.get(id, None)
    
    def _update_index(self, id, vector):
        # Ensure vector_index is updated with cosine similarities
        for existing_id, existing_vector in self.vectors.items():
            if existing_id == id:
                continue  
            similarity = np.dot(vector, existing_vector) / (np.linalg.norm(vector) * np.linalg.norm(existing_vector))
            if existing_id not in self.vector_index:
                self.vector_index[existing_id] = {}
            self.vector_index[existing_id][id] = similarity

    def search(self, query_vector, top_k=5):
        # Perform similarity search
        query_vector = np.array(query_vector)  # Ensure query_vector is a numpy array
        distances = []
        for id, vector in self.vectors.items():
            dist = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            distances.append((id, dist))

        distances.sort(key=lambda x: x[1], reverse=True) # Sort by cosine similarity
        return distances[:top_k]

