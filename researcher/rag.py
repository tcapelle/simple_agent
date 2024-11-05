from typing import List, Dict
import openai
import numpy as np
from scipy.spatial.distance import cdist


class DenseRetriever:
    """
    A retriever model that uses OpenAI embeddings for indexing and searching documents.

    Attributes:
        index (np.ndarray): The indexed embeddings.
        data (list): The data to be indexed.
    """

    def __init__(self):
        self.index = None
        self.data = None

    def index_data(self, data: List[Dict[str, str]]) -> None:
        """
        Indexes the provided data using OpenAI embeddings.

        Args:
            data (list): A list of documents to be indexed. Each document should be a dictionary
                         containing a key 'content' with the text to be indexed.
        """
        self.data = data
        texts = [doc["content"] for doc in data]
        embeddings = self.get_embeddings(texts)
        self.index = np.array(embeddings)

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of texts using OpenAI's embedding API.

        Args:
            texts (list): A list of text strings to generate embeddings for.

        Returns:
            list: A list of embeddings.
        """
        client = openai.OpenAI()
        
        response = client.embeddings.create(input=texts, model="text-embedding-3-small")
        return [embed.embedding for embed in response.data]

    def search(self, query: str, k: int = 5) -> List[Dict[str, str]]:
        """
        Searches the indexed data for the given query using cosine similarity.

        Args:
            query (str): The search query.
            k (int): The number of top results to return. Default is 5.

        Returns:
            list: A list of dictionaries containing the source, text, and score of the top-k results.
        """
        query_embedding = self.get_embeddings([query])[0]
        cosine_distances = cdist([query_embedding], self.index, metric="cosine")[0]
        top_k_indices = cosine_distances.argsort()[:k]
        output = []
        for idx in top_k_indices:
            output.append(
                {
                    "source": self.data[idx].get("source", "unknown"),
                    "text": self.data[idx]["content"],
                    "score": 1 - cosine_distances[idx],
                }
            )
        return output
