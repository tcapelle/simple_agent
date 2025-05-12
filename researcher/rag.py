import os
import pickle
import json
import numpy as np
from typing import List, Dict, Any
from tqdm.asyncio import tqdm_asyncio
import threading
from pathlib import Path
from dataclasses import dataclass
import simple_parsing as sp
import asyncio
from mistralai import Mistral

import weave

from researcher.config import DATA_DIR, DEFAULT_EMBEDDING_MODEL, DEFAULT_MODEL, PARALLEL_REQUESTS, DEFAULT_WEAVE_PROJECT
from researcher.console import console

@dataclass
class RAGArgs:
    """Arguments for RAG processing"""
    data_dir: Path = sp.field(
        default=DATA_DIR, 
        help="Path to store/load the vector database"
    )
    model: str = sp.field(
        default=DEFAULT_MODEL,
        help="Mistral model to use for context generation"
    )
    embedding_model: str = sp.field(
        default=DEFAULT_EMBEDDING_MODEL,
        help="Model to use for embeddings"
    )
    temperature: float = sp.field(
        default=0.0,
        help="Temperature for context generation"
    )
    max_tokens: int = sp.field(
        default=1000,
        help="Maximum tokens for context generation"
    )
    parallel_requests: int = sp.field(
        default=PARALLEL_REQUESTS,
        help="Number of parallel requests for processing"
    )
    debug: bool = sp.field(
        default=False,
        help="Debug mode. Only process the first 2 documents"
    )
    weave_project: str = sp.field(
        default=DEFAULT_WEAVE_PROJECT,
        help="Weave project name"
    )

class ContextualVectorDB:
    def __init__(self, 
                 db_path: Path = Path("./my_data"),
                 model: str = "gpt-4o",
                 embedding_model: str = "mistral-embed",
                 temperature: float = 0.0,
                 max_tokens: int = 1000,
                 **kwargs):
        
        # Initialize Mistral client for embeddings
        self.mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        self.embeddings = []
        self.metadata = []
        self.query_cache = {}
        self.db_path = db_path / "contextual_vector_db.pkl"
        
        # Store configuration
        self.config = {
            'model': model,
            'embedding_model': embedding_model,
            'temperature': temperature,
            'max_tokens': max_tokens
        }

        self.token_counts = {
            'input': 0,
            'output': 0,
            'cache_read': 0,
            'cache_creation': 0
        }
        self.token_lock = threading.Lock()

    @weave.op
    async def situate_context(self, doc: str, chunk: str) -> tuple[str, Any]:
        SYSTEM_PROMPT = """You will be given a document and a chunk from that document. Your task is to provide a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""
        
        USER_PROMPT = f"""
        Document:
        {doc}

        Chunk to situate:
        {chunk}
        """

        try:
            response = await self.mistral_client.chat.complete_async(
                model=self.config['model'],
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT}
                ],
                max_tokens=self.config['max_tokens'],
                temperature=self.config['temperature']
            )
            
            with self.token_lock:
                # Track token usage from Mistral
                input_tokens = getattr(response.usage, 'prompt_tokens', 0)
                output_tokens = getattr(response.usage, 'completion_tokens', 0)
                self.token_counts['input'] += input_tokens
                self.token_counts['output'] += output_tokens
                
            return response.choices[0].message.content, response.usage
        except Exception as e:
            console.print(f"[red]Error in situate_context: {str(e)}")
            raise

    async def load_data(self, dataset: List[Dict[str, Any]], parallel_requests: int = 25):
        if self.embeddings and self.metadata:
            print("Vector database is already loaded. Skipping data loading.")
            return
        if os.path.exists(self.db_path):
            print(f"Loading vector database from disk: {self.db_path}")
            return self.load_db(self.db_path)

        texts_to_embed = []
        metadata = []

        # Process chunks using semaphore for parallel requests
        semaphore = asyncio.Semaphore(parallel_requests)
        tasks = []
        
        async def process_chunk(doc, chunk):
            async with semaphore:
                return await self.situate_context(doc['content'], chunk['content'])
        
        for doc in dataset:
            for chunk in doc['chunks']:
                tasks.append(process_chunk(doc, chunk))
        
        # Use tqdm_asyncio.gather directly without manual event loop management
        results = await tqdm_asyncio.gather(*tasks)
        
        all_chunks = [(doc, chunk) 
                     for doc in dataset 
                     for chunk in doc['chunks']]
        
        for (doc, chunk), (contextualized_text, usage) in zip(all_chunks, results):
            texts_to_embed.append(f"{chunk['content']}\n\n{contextualized_text}")
            metadata.append({
                'doc_id': doc['doc_id'],
                'original_uuid': doc['original_uuid'],
                'chunk_id': chunk['chunk_id'],
                'original_index': chunk['original_index'],
                'original_content': chunk['content'],
                'contextualized_content': contextualized_text
            })

        await self._embed_and_store(texts_to_embed, metadata)
        self.save_db()

        # logging token usage
        print(f"Contextual Vector database loaded and saved. Total chunks processed: {len(texts_to_embed)}")
        print(f"Total input tokens without caching: {self.token_counts['input']}")
        print(f"Total output tokens: {self.token_counts['output']}")
        print(f"Total input tokens written to cache: {self.token_counts['cache_creation']}")
        print(f"Total input tokens read from cache: {self.token_counts['cache_read']}")
        
        total_tokens = self.token_counts['input'] + self.token_counts['cache_read'] + self.token_counts['cache_creation']
        savings_percentage = (self.token_counts['cache_read'] / total_tokens) * 100 if total_tokens > 0 else 0
        print(f"Total input token savings from prompt caching: {savings_percentage:.2f}% of all input tokens used were read from cache.")
        print("Tokens read from cache come at a 90 percent discount!")

    async def _embed_and_store(self, texts: List[str], data: List[Dict[str, Any]]):
        batch_size = 128
        all_embeddings = []
        
        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                # Use Mistral for batch embeddings
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.mistral_client.embeddings.create(
                        model=self.config['embedding_model'],
                        inputs=batch
                    )
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
            self.embeddings = all_embeddings
            self.metadata = data
        except Exception as e:
            console.print(f"[red]Error in _embed_and_store: {str(e)}")
            raise

    def search(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        """Synchronous version of search"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.asearch(query, k))
        finally:
            loop.close()

    @weave.op
    async def asearch(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        if query in self.query_cache:
            query_embedding = self.query_cache[query]
        else:
            # Use Mistral for query embedding
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.mistral_client.embeddings.create(
                    model=self.config['embedding_model'],
                    inputs=[query]
                )
            )
            query_embedding = response.data[0].embedding
            self.query_cache[query] = query_embedding

        if not self.embeddings:
            raise ValueError("No data loaded in the vector database.")

        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1][:k]
        
        top_results = []
        for idx in top_indices:
            result = {
                "metadata": self.metadata[idx],
                "similarity": float(similarities[idx]),
            }
            top_results.append(result)
        return top_results

    def save_db(self):
        data = {
            "embeddings": self.embeddings,
            "metadata": self.metadata,
            "query_cache": json.dumps(self.query_cache),
            "config": self.config,
        }
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "wb") as file:
            pickle.dump(data, file)

    @classmethod
    def load_db(cls, db_path: Path) -> 'ContextualVectorDB':
        if not os.path.exists(db_path):
            raise ValueError("Vector database file not found. Use load_data to create a new database.")
        
        with open(db_path, "rb") as file:
            data = pickle.load(file)
        
        # Use config from file if available, otherwise use default values
        default_config = {
            'model': "mistral-medium-latest",
            'embedding_model': "mistral-embed",
            'temperature': 0.0,
            'max_tokens': 1000
        }
        config = data.get("config", default_config)
        
        # Create instance with config
        instance = cls(
            db_path=Path(os.path.dirname(db_path)),
            **config
        )
        
        instance.embeddings = data["embeddings"]
        instance.metadata = data["metadata"]
        instance.query_cache = json.loads(data["query_cache"])
        
        return instance


def create_db():
    args = sp.parse(RAGArgs)

    weave.init(args.weave_project)
    
    @weave.op
    async def process_dataset():
        try:
            # Load the transformed dataset
            transformed_dataset = []
            with open(f'{args.data_dir}/processed_documents.jsonl', 'r') as f:
                for i, line in enumerate(f):
                    if args.debug and i > 1:
                        break
                    transformed_dataset.append(json.loads(line))
            
            db = ContextualVectorDB(
                db_path=args.data_dir,
                model=args.model,
                embedding_model=args.embedding_model,
                temperature=args.temperature,
                max_tokens=args.max_tokens
            )
            
            # Test single context generation
            console.print("[green]Testing single context generation...")
            result = await db.situate_context(
                transformed_dataset[0]['content'], 
                transformed_dataset[0]['chunks'][0]['content']
            )
            console.print(f"Sample context: {result[0]}")
            
            # Load all data
            console.print("[green]Loading full dataset...")
            db = await db.load_data(transformed_dataset, parallel_requests=args.parallel_requests)
            return db
        except Exception as e:
            console.print(f"[red]Error in main process: {str(e)}")
            raise

    # Run the async main function
    db = asyncio.run(process_dataset())

    # Test search functionality
    console.print("[green]Testing search functionality...")
    sample_query = "What Unesco places are in Chile?"  # Replace with an actual test query
    results = db.search(sample_query, k=3)
    console.print(f"Sample search results: {results}")
            

def prepare():
    from researcher.preprocess import main
    main()
    console.rule("[green]Creating vector database...")
    create_db()

if __name__ == "__main__":
    create_db()
