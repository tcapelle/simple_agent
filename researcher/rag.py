import os
import pickle
import json
import numpy as np
import openai
import anthropic
from typing import List, Dict, Any
from tqdm.asyncio import tqdm_asyncio
import threading
from pathlib import Path
from dataclasses import dataclass
import simple_parsing as sp
import asyncio

import weave

from researcher.console import console

@dataclass
class RAGArgs:
    """Arguments for RAG processing"""
    db_path: Path = sp.field(
        default=Path("./my_data"), 
        help="Path to store/load the vector database"
    )
    openai_api_key: str = sp.field(
        default=None, 
        help="OpenAI API key. Defaults to OPENAI_API_KEY environment variable"
    )
    model: str = sp.field(
        default="gpt-4o",
        help="OpenAI model to use for context generation"
    )
    embedding_model: str = sp.field(
        default="text-embedding-3-small",
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
        default=5,
        help="Number of parallel requests for processing"
    )
    debug: bool = sp.field(
        default=False,
        help="Debug mode. Only process the first 2 documents"
    )
    anthropic_api_key: str = sp.field(
        default=None, 
        help="Anthropic API key. Defaults to ANTHROPIC_API_KEY environment variable"
    )
    claude_model: str = sp.field(
        default="claude-3-5-haiku-latest",
        help="Anthropic Claude model to use for context generation"
    )
    weave_project: str = sp.field(
        default="researcher",
        help="Weave project name"
    )

class ContextualVectorDB:
    def __init__(self, args: RAGArgs):
        if args.openai_api_key is None:
            args.openai_api_key = os.getenv("OPENAI_API_KEY")
        if args.anthropic_api_key is None:
            args.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        
        self.client = openai.AsyncOpenAI(api_key=args.openai_api_key)
        self.anthropic_client = anthropic.AsyncAnthropic(api_key=args.anthropic_api_key)
        self.embeddings = []
        self.metadata = []
        self.query_cache = {}
        self.db_path = args.db_path / "contextual_vector_db.pkl"
        self.args = args

        self.token_counts = {
            'input': 0,
            'output': 0,
            'cache_read': 0,
            'cache_creation': 0
        }
        self.token_lock = threading.Lock()

    @weave.op()
    async def situate_context(self, doc: str, chunk: str) -> tuple[str, Any]:
        DOCUMENT_CONTEXT_PROMPT = """
        <document>
        {doc_content}
        </document>
        """

        CHUNK_CONTEXT_PROMPT = """
        Here is the chunk we want to situate within the whole document
        <chunk>
        {chunk_content}
        </chunk>

        Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
        Answer only with the succinct context and nothing else.
        """

        try:
            response = await self.anthropic_client.beta.prompt_caching.messages.create(
                model=self.args.claude_model,
                max_tokens=self.args.max_tokens,
                temperature=self.args.temperature,
                messages=[
                    {
                        "role": "user", 
                        "content": [
                            {
                                "type": "text",
                                "text": DOCUMENT_CONTEXT_PROMPT.format(doc_content=doc),
                                "cache_control": {"type": "ephemeral"}
                            },
                            {
                                "type": "text",
                                "text": CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk),
                            },
                        ]
                    },
                ],
                extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
            )
            
            with self.token_lock:
                self.token_counts['input'] += response.usage.input_tokens
                self.token_counts['output'] += response.usage.output_tokens
                self.token_counts['cache_read'] += response.usage.cache_read_input_tokens
                self.token_counts['cache_creation'] += response.usage.cache_creation_input_tokens
            
            return response.content[0].text, response.usage
        except Exception as e:
            console.print(f"[red]Error in situate_context: {str(e)}")
            raise

    async def load_data(self, dataset: List[Dict[str, Any]], parallel_requests: int = 25):
        if self.embeddings and self.metadata:
            print("Vector database is already loaded. Skipping data loading.")
            return
        if os.path.exists(self.db_path):
            print("Loading vector database from disk.")
            self.load_db()
            return

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

    #we use voyage AI here for embeddings. Read more here: https://docs.voyageai.com/docs/embeddings
    async def _embed_and_store(self, texts: List[str], data: List[Dict[str, Any]]):
        batch_size = 128
        all_embeddings = []
        
        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                response = await self.client.embeddings.create(
                    model=self.args.embedding_model,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
            self.embeddings = all_embeddings
            self.metadata = data
        except Exception as e:
            console.print(f"[red]Error in _embed_and_store: {str(e)}")
            raise

    @weave.op
    async def search(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        if query in self.query_cache:
            query_embedding = self.query_cache[query]
        else:
            # Replace Voyage query embedding with OpenAI
            response = await self.client.embeddings.create(
                model=self.args.embedding_model,
                input=[query]
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
        }
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "wb") as file:
            pickle.dump(data, file)

    def load_db(self):
        if not os.path.exists(self.db_path):
            raise ValueError("Vector database file not found. Use load_data to create a new database.")
        with open(self.db_path, "rb") as file:
            data = pickle.load(file)
        self.embeddings = data["embeddings"]
        self.metadata = data["metadata"]
        self.query_cache = json.loads(data["query_cache"])


if __name__ == "__main__":
    args = sp.parse(RAGArgs)

    weave.init(args.weave_project)
    
    @weave.op
    async def process_dataset():
        try:
            # Load the transformed dataset
            transformed_dataset = []
            with open('my_data/processed_documents.jsonl', 'r') as f:
                for i, line in enumerate(f):
                    if args.debug and i > 1:
                        break
                    transformed_dataset.append(json.loads(line))
            
            db = ContextualVectorDB(args)
            
            # Test single context generation
            console.print("[green]Testing single context generation...")
            result = await db.situate_context(
                transformed_dataset[0]['content'], 
                transformed_dataset[0]['chunks'][0]['content']
            )
            console.print(f"Sample context: {result[0]}")
            
            # Load all data
            console.print("[green]Loading full dataset...")
            await db.load_data(transformed_dataset, parallel_requests=args.parallel_requests)
            
            # Test search functionality
            console.print("[green]Testing search functionality...")
            sample_query = "test query"  # Replace with an actual test query
            results = await db.search(sample_query, k=3)
            console.print(f"Sample search results: {results}")
            
        except Exception as e:
            console.print(f"[red]Error in main process: {str(e)}")
            raise

    # Run the async main function
    asyncio.run(process_dataset())
