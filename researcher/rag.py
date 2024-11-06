from typing import List, Dict
import openai
import chromadb
from chromadb.utils import embedding_functions
import os

class DenseRetriever:
    """
    A retriever model that uses OpenAI embeddings for indexing and searching documents.

    Attributes:
        client (chromadb.Client): The ChromaDB client
        collection (chromadb.Collection): The ChromaDB collection for storing documents
        openai_ef (embedding_functions.OpenAIEmbeddingFunction): OpenAI embedding function
    """

    def __init__(self, collection_name: str = "research_documents"):
        """
        Initialize the retriever with ChromaDB and OpenAI embeddings.

        Args:
            collection_name (str): Name of the ChromaDB collection
        """
        # Initialize ChromaDB client
        self.client = chromadb.Client()
        
        # Initialize OpenAI embedding function
        self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ["OPENAI_API_KEY"],
            model_name="text-embedding-ada-002"
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.openai_ef
        )

    def index_data(self, data: List[Dict[str, str]]) -> None:
        """
        Indexes the provided data using ChromaDB.

        Args:
            data (list): A list of documents to be indexed. Each document should be a dictionary
                        containing 'content' and optionally 'source'.
        """
        documents = []
        ids = []
        metadata = []
        
        for idx, doc in enumerate(data):
            documents.append(doc["content"])
            ids.append(f"doc_{idx}")
            metadata.append({
                "source": doc.get("source", "unknown"),
                "content": doc["content"]
            })
        
        # Upsert documents to collection
        self.collection.upsert(
            documents=documents,
            ids=ids,
            metadatas=metadata
        )
        
        print(f"Indexed {len(documents)} documents")

    def search(self, query: str, k: int = 5) -> List[Dict[str, str]]:
        """
        Searches the indexed data for the given query.

        Args:
            query (str): The search query
            k (int): Number of results to return (default: 5)

        Returns:
            list: List of dictionaries containing search results with source, text, and score
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        
        output = []
        if results['ids'][0]:  # Check if we have any results
            for i in range(len(results['ids'][0])):
                output.append({
                    "source": results['metadatas'][0][i].get("source", "unknown"),
                    "text": results['metadatas'][0][i]["content"],
                    "score": 1 - (results['distances'][0][i] if results['distances'] else 0)
                })
        
        return output

    def clear_index(self):
        """
        Clears all documents from the collection.
        """
        self.collection.delete(where={})
        print("Cleared all documents from the collection")
