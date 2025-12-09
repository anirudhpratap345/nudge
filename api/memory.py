"""
Memory module: FAISS for vector search + Redis for recent messages.

Vercel (Hobby) cannot build heavy deps like faiss-cpu/sentence-transformers
within the 8 GB build box. To keep the API running on Groq-only deployments,
we gracefully fall back to a lightweight in-memory store when FAISS/embeddings
are not installed. Long-term vector memory will be disabled in that case, but
the API and Groq chat will still work.
"""
from typing import List, Optional, Dict, Any
import json
import os
import pickle
from datetime import datetime
import logging
import redis

from config import get_settings

logger = logging.getLogger(__name__)

# Optional heavy deps ---------------------------------------------------------
FAISS_AVAILABLE = True
try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None  # type: ignore
    np = None     # type: ignore
    SentenceTransformer = None  # type: ignore
    logger.warning(
        "FAISS/sentence-transformers not installed. "
        "Running in Groq-only mode with no vector memory."
    )


if FAISS_AVAILABLE:
    class EmbeddingModel:
        """
        Embedding function using sentence-transformers
        Tries NV-Embed-v2 first, falls back to all-MiniLM-L6-v2
        """
        
        def __init__(self, model_name: str = "nvidia/NV-Embed-v2", device: str = "cuda"):
            self.model_name = model_name
            self.device = device
            self._model = None
            self._dimension = None
        
        @property
        def model(self):
            """Lazy load the embedding model"""
            if self._model is None:
                logger.info(f"Loading embedding model: {self.model_name}")
                try:
                    # Try the specified model first
                    self._model = SentenceTransformer(
                        self.model_name,
                        device=self.device,
                        trust_remote_code=True
                    )
                    logger.info(f"✅ Loaded {self.model_name}")
                except Exception as e:
                    # Fallback to a lighter model
                    logger.warning(f"Failed to load {self.model_name}: {e}")
                    logger.info("Falling back to all-MiniLM-L6-v2")
                    self._model = SentenceTransformer(
                        "all-MiniLM-L6-v2",
                        device="cpu"  # This small model runs fine on CPU
                    )
            return self._model
        
        @property
        def dimension(self) -> int:
            """Get embedding dimension"""
            if self._dimension is None:
                # Encode a test sentence to get dimension
                test_embedding = self.encode(["test"])
                self._dimension = test_embedding.shape[1]
            return self._dimension
        
        def encode(self, texts: List[str]) -> np.ndarray:
            """Generate embeddings for a list of texts"""
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings.astype('float32')  # FAISS requires float32


    class FAISSMemoryStore:
        """
        FAISS-based vector store for long-term memories
        Persists to disk for durability
        """
        
        def __init__(self, persist_dir: str, dimension: int):
            self.persist_dir = persist_dir
            self.dimension = dimension
            
            # Create persist directory
            os.makedirs(persist_dir, exist_ok=True)
            
            # File paths
            self.index_path = os.path.join(persist_dir, "faiss.index")
            self.metadata_path = os.path.join(persist_dir, "metadata.pkl")
            
            # Initialize or load index
            self._index = None
            self._metadata: List[Dict[str, Any]] = []
            self._load_or_create()
        
        def _load_or_create(self):
            """Load existing index or create new one"""
            if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
                logger.info(f"Loading FAISS index from {self.persist_dir}")
                self._index = faiss.read_index(self.index_path)
                with open(self.metadata_path, 'rb') as f:
                    self._metadata = pickle.load(f)
                logger.info(f"Loaded {self._index.ntotal} vectors")
            else:
                logger.info(f"Creating new FAISS index (dim={self.dimension})")
                # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
                self._index = faiss.IndexFlatIP(self.dimension)
                self._metadata = []
        
        def _save(self):
            """Persist index to disk"""
            faiss.write_index(self._index, self.index_path)
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self._metadata, f)
        
        def add(self, embeddings: np.ndarray, metadata_list: List[Dict[str, Any]]):
            """Add vectors with metadata"""
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add to index
            self._index.add(embeddings)
            self._metadata.extend(metadata_list)
            
            # Persist
            self._save()
        
        def search(
            self,
            query_embedding: np.ndarray,
            k: int = 8,
            filter_fn: Optional[callable] = None
        ) -> List[Dict[str, Any]]:
            """Search for similar vectors"""
            if self._index.ntotal == 0:
                return []
            
            # Normalize query
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            faiss.normalize_L2(query_embedding)
            
            # Search more than k if we need to filter
            search_k = min(k * 3, self._index.ntotal) if filter_fn else min(k, self._index.ntotal)
            
            distances, indices = self._index.search(query_embedding, search_k)
            
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for empty slots
                    continue
                
                meta = self._metadata[idx].copy()
                meta['score'] = float(dist)
                
                # Apply filter if provided
                if filter_fn and not filter_fn(meta):
                    continue
                
                results.append(meta)
                
                if len(results) >= k:
                    break
            
            return results
        
        def delete_by_user(self, user_id: str) -> int:
            """Delete all vectors for a user (requires rebuilding index)"""
            # Find indices to keep
            keep_indices = []
            keep_metadata = []
            
            for i, meta in enumerate(self._metadata):
                if meta.get('user_id') != user_id:
                    keep_indices.append(i)
                    keep_metadata.append(meta)
            
            deleted_count = len(self._metadata) - len(keep_metadata)
            
            if deleted_count > 0:
                # Rebuild index with remaining vectors
                if keep_indices:
                    # Get vectors to keep
                    vectors = np.zeros((len(keep_indices), self.dimension), dtype='float32')
                    for new_idx, old_idx in enumerate(keep_indices):
                        vectors[new_idx] = self._index.reconstruct(old_idx)
                    
                    # Create new index
                    self._index = faiss.IndexFlatIP(self.dimension)
                    self._index.add(vectors)
                else:
                    self._index = faiss.IndexFlatIP(self.dimension)
                
                self._metadata = keep_metadata
                self._save()
            
            return deleted_count
        
        def get_user_count(self, user_id: str) -> int:
            """Count memories for a user"""
            return sum(1 for m in self._metadata if m.get('user_id') == user_id)
else:
    class EmbeddingModel:
        """Lightweight stub when embeddings are not available."""
        def __init__(self, *args, **kwargs):
            self.model_name = "stub"
            self.device = "cpu"
            self._dimension = 0

        @property
        def dimension(self) -> int:
            return 0

        def encode(self, texts: List[str]):
            return []


    class FAISSMemoryStore:
        """No-op vector store stub."""
        def __init__(self, persist_dir: str, dimension: int):
            self._metadata: List[Dict[str, Any]] = []

        def add(self, embeddings, metadata_list: List[Dict[str, Any]]):
            self._metadata.extend(metadata_list)

        def search(self, query_embedding=None, k: int = 8, filter_fn: Optional[callable] = None) -> List[Dict[str, Any]]:
            # Return nothing to avoid misleading context
            return []

        def delete_by_user(self, user_id: str) -> int:
            before = len(self._metadata)
            self._metadata = [m for m in self._metadata if m.get("user_id") != user_id]
            return before - len(self._metadata)

        def get_user_count(self, user_id: str) -> int:
            return sum(1 for m in self._metadata if m.get('user_id') == user_id)


class InMemoryCache:
    """Simple in-memory cache when Redis is not available"""
    
    def __init__(self, max_messages: int = 10):
        self.max_messages = max_messages
        self._cache: Dict[str, List[Dict[str, str]]] = {}
    
    def add_message(self, user_id: str, role: str, content: str):
        if user_id not in self._cache:
            self._cache[user_id] = []
        
        self._cache[user_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Keep only last N messages
        if len(self._cache[user_id]) > self.max_messages:
            self._cache[user_id] = self._cache[user_id][-self.max_messages:]
    
    def get_messages(self, user_id: str) -> List[Dict[str, str]]:
        return self._cache.get(user_id, [])
    
    def clear_user(self, user_id: str):
        if user_id in self._cache:
            del self._cache[user_id]


class MemoryManager:
    """
    Manages long-term memory (FAISS) and short-term cache (Redis or in-memory)
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._embed_model = None
        self._faiss_store = None
        self._redis_client = None
        self._use_memory_cache = False
        self._memory_cache = InMemoryCache(max_messages=self.settings.redis_cache_size)
        self._fallback_memories: Dict[str, List[Dict[str, Any]]] = {}
    
    @property
    def embed_model(self) -> EmbeddingModel:
        """Get embedding model"""
        if self._embed_model is None:
            self._embed_model = EmbeddingModel(
                model_name=self.settings.embedding_model,
                device=self.settings.embedding_device
            )
        return self._embed_model
    
    @property
    def faiss_store(self) -> FAISSMemoryStore:
        """Get FAISS store"""
        if self._faiss_store is None:
            self._faiss_store = FAISSMemoryStore(
                persist_dir=self.settings.chroma_persist_dir,  # Reusing same config
                dimension=self.embed_model.dimension
            )
        return self._faiss_store
    
    @property
    def redis_client(self) -> Optional[redis.Redis]:
        """Get Redis client for recent message cache"""
        if self._redis_client is None and self.settings.redis_enabled:
            try:
                # Support both local Redis and Upstash (cloud)
                if self.settings.redis_ssl or "upstash" in self.settings.redis_url.lower():
                    # Upstash requires SSL
                    self._redis_client = redis.from_url(
                        self.settings.redis_url,
                        decode_responses=True,
                        ssl_cert_reqs=None  # Skip cert verification for Upstash
                    )
                else:
                    # Local Redis/Memurai
                    self._redis_client = redis.from_url(
                        self.settings.redis_url,
                        decode_responses=True,
                        password=self.settings.redis_password
                    )
                self._redis_client.ping()
                logger.info("✅ Redis connected")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Using in-memory fallback.")
                self._redis_client = None
                self._use_memory_cache = True
        return self._redis_client
    
    def store_memory(
        self,
        user_id: str,
        content: str,
        memory_type: str = "conversation",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a memory in FAISS
        
        Args:
            user_id: Unique user identifier
            content: The memory content to store
            memory_type: Type of memory (conversation, goal, win, struggle, etc.)
            metadata: Additional metadata
        
        Returns:
            memory_id: The ID of the stored memory
        """
        timestamp = datetime.utcnow().isoformat()
        memory_id = f"{user_id}_{timestamp}_{memory_type}"
        
        # Prepare metadata
        doc_metadata = {
            "id": memory_id,
            "user_id": user_id,
            "content": content,
            "memory_type": memory_type,
            "timestamp": timestamp,
            **(metadata or {})
        }
        if FAISS_AVAILABLE:
            # Generate embedding
            embedding = self.embed_model.encode([content])
            
            # Store in FAISS
            self.faiss_store.add(embedding, [doc_metadata])
            
            logger.info(f"Stored memory {memory_id} for user {user_id}")
        else:
            # Lightweight fallback: keep last N memories per user in-memory
            self._fallback_memories.setdefault(user_id, [])
            self._fallback_memories[user_id].append(doc_metadata)
            # Keep it bounded
            self._fallback_memories[user_id] = self._fallback_memories[user_id][-self.settings.memory_top_k :]
            logger.info(f"Stored fallback memory {memory_id} for user {user_id}")
        return memory_id
    
    def retrieve_memories(
        self,
        user_id: str,
        query: str,
        n_results: int = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories for a user based on query
        
        This is the key retrieval function called before every LLM response:
        retrieved = memory.retrieve_memories(user_id, user_message, n_results=8)
        
        Args:
            user_id: Unique user identifier
            query: The query to search for (usually the user's message)
            n_results: Number of results to return
        
        Returns:
            List of memory documents with metadata
        """
        n_results = n_results or self.settings.memory_top_k
        # If FAISS is unavailable (Vercel slim build), return fallback memories
        if not FAISS_AVAILABLE:
            fallback = self._fallback_memories.get(user_id, [])
            return [
                {
                    "content": mem.get("content", ""),
                    "metadata": {
                        "memory_type": mem.get("memory_type", "unknown"),
                        "timestamp": mem.get("timestamp", ""),
                        "user_id": mem.get("user_id", "")
                    },
                    "score": 0
                }
                for mem in fallback[-n_results:]
            ]
        
        # Generate query embedding
        query_embedding = self.embed_model.encode([query])
        
        # Search FAISS with user filter
        results = self.faiss_store.search(
            query_embedding[0],
            k=n_results,
            filter_fn=lambda m: m.get('user_id') == user_id
        )
        
        # Format results
        memories = []
        for result in results:
            memories.append({
                "content": result.get("content", ""),
                "metadata": {
                    "memory_type": result.get("memory_type", "unknown"),
                    "timestamp": result.get("timestamp", ""),
                    "user_id": result.get("user_id", "")
                },
                "score": result.get("score", 0)
            })
        
        logger.info(f"Retrieved {len(memories)} memories for user {user_id}")
        return memories
    
    def format_memory_context(self, memories: List[Dict[str, Any]]) -> str:
        """
        Format retrieved memories into a context string for the LLM
        
        Args:
            memories: List of memory documents
        
        Returns:
            Formatted string of memories
        """
        if not memories:
            return ""
        
        memory_lines = []
        for mem in memories:
            meta = mem.get("metadata", {})
            mem_type = meta.get("memory_type", "memory")
            timestamp = meta.get("timestamp", "")
            content = mem.get("content", "")
            
            if timestamp:
                # Format: [win, 2025-12-01] User shipped Redis caching...
                date_part = timestamp.split("T")[0] if "T" in timestamp else timestamp
                memory_lines.append(f"[{mem_type}, {date_part}] {content}")
            else:
                memory_lines.append(f"[{mem_type}] {content}")
        
        return "\n".join(memory_lines)
    
    # =========================================
    # Message Cache (Redis or In-Memory)
    # =========================================
    
    def cache_message(self, user_id: str, role: str, content: str):
        """Cache a message for quick access (Redis or in-memory fallback)"""
        # Try Redis first
        if self.redis_client:
            try:
                key = f"nudge:history:{user_id}"
                message = json.dumps({
                    "role": role,
                    "content": content,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                # Add to list and trim to last N messages
                self.redis_client.rpush(key, message)
                self.redis_client.ltrim(key, -self.settings.redis_cache_size, -1)
                
                # Set expiry (7 days)
                self.redis_client.expire(key, 60 * 60 * 24 * 7)
                return
            except Exception as e:
                logger.warning(f"Redis cache_message failed: {e}")
                self._use_memory_cache = True
        
        # Fallback to in-memory cache
        self._memory_cache.add_message(user_id, role, content)
    
    def get_recent_messages(self, user_id: str) -> List[Dict[str, str]]:
        """Get recent messages from cache (Redis or in-memory fallback)"""
        # Try Redis first
        if self.redis_client and not self._use_memory_cache:
            try:
                key = f"nudge:history:{user_id}"
                messages = self.redis_client.lrange(key, 0, -1)
                return [json.loads(msg) for msg in messages]
            except Exception as e:
                logger.warning(f"Redis get_recent_messages failed: {e}")
                self._use_memory_cache = True
        
        # Fallback to in-memory cache
        return self._memory_cache.get_messages(user_id)
    
    def format_conversation_history(self, messages: List[Dict[str, str]]) -> str:
        """Format conversation history for the prompt"""
        if not messages:
            return ""
        
        lines = []
        for msg in messages:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        
        return "\n".join(lines)
    
    # =========================================
    # Utility Methods
    # =========================================
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get stats about a user's stored memories"""
        if not FAISS_AVAILABLE:
            user_mems = self._fallback_memories.get(user_id, [])
            memory_types: Dict[str, int] = {}
            for mem in user_mems:
                mem_type = mem.get("memory_type", "unknown")
                memory_types[mem_type] = memory_types.get(mem_type, 0) + 1
            return {
                "user_id": user_id,
                "total_memories": len(user_mems),
                "memory_types": memory_types
            }
        # Get all memories for user to count types
        all_memories = self.faiss_store.search(
            self.embed_model.encode(["user memories"])[0],
            k=1000,
            filter_fn=lambda m: m.get('user_id') == user_id
        )
        
        memory_types = {}
        for mem in all_memories:
            mem_type = mem.get("memory_type", "unknown")
            memory_types[mem_type] = memory_types.get(mem_type, 0) + 1
        
        return {
            "user_id": user_id,
            "total_memories": len(all_memories),
            "memory_types": memory_types
        }
    
    def delete_user_memories(self, user_id: str) -> int:
        """Delete all memories for a user (GDPR compliance)"""
        if not FAISS_AVAILABLE:
            before = len(self._fallback_memories.get(user_id, []))
            self._fallback_memories[user_id] = []
            count = before
        else:
            count = self.faiss_store.delete_by_user(user_id)
        
        # Clear Redis cache
        if self.redis_client:
            try:
                self.redis_client.delete(f"nudge:history:{user_id}")
            except Exception as e:
                logger.warning(f"Redis delete failed: {e}")
        
        # Also clear in-memory cache
        self._memory_cache.clear_user(user_id)
        
        return count


# Global memory manager instance
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Get the global memory manager instance"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager
