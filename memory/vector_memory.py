"""
Vector Memory - Optional semantic search using FAISS or ChromaDB.
Falls back gracefully if dependencies are not installed.
"""

from typing import List, Dict, Optional, Any
from core.logger import get_logger

logger = get_logger(__name__)

# Try to import vector libraries
try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False


class VectorMemory:
    """
    Vector-based semantic memory for finding similar past interactions.
    Requires sentence-transformers and either FAISS or ChromaDB.

    Install: pip install sentence-transformers faiss-cpu chromadb
    """

    def __init__(self, backend: str = "faiss", persist_dir: str = "data/vectors") -> None:
        self.backend = backend
        self.persist_dir = persist_dir
        self._ready = False
        self._embedder = None
        self._index = None
        self._documents: List[Dict] = []

    def initialize(self) -> bool:
        """Initialize the vector store. Returns True if successful."""
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Sentence transformer loaded")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Vector memory disabled. Run: pip install sentence-transformers"
            )
            return False

        if self.backend == "faiss":
            return self._init_faiss()
        elif self.backend == "chroma":
            return self._init_chroma()
        else:
            logger.warning(f"Unknown vector backend: {self.backend}")
            return False

    def _init_faiss(self) -> bool:
        """Initialize FAISS index."""
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available. Run: pip install faiss-cpu")
            return False
        try:
            # 384 dimensions for all-MiniLM-L6-v2
            self._index = faiss.IndexFlatL2(384)
            self._ready = True
            logger.info("FAISS vector index initialized")
            return True
        except Exception as e:
            logger.error(f"FAISS init failed: {e}")
            return False

    def _init_chroma(self) -> bool:
        """Initialize ChromaDB."""
        if not CHROMA_AVAILABLE:
            logger.warning("ChromaDB not available. Run: pip install chromadb")
            return False
        try:
            import chromadb
            from pathlib import Path
            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=self.persist_dir)
            self._index = client.get_or_create_collection("jarvis_memory")
            self._ready = True
            logger.info("ChromaDB vector store initialized")
            return True
        except Exception as e:
            logger.error(f"ChromaDB init failed: {e}")
            return False

    def add(self, text: str, metadata: Optional[Dict] = None) -> bool:
        """Add a document to vector memory."""
        if not self._ready:
            return False
        try:
            if self.backend == "faiss":
                embedding = self._embedder.encode([text])
                self._index.add(embedding.astype("float32"))
                self._documents.append({"text": text, "metadata": metadata or {}})
            elif self.backend == "chroma":
                doc_id = str(len(self._documents))
                self._index.add(
                    documents=[text],
                    metadatas=[metadata or {}],
                    ids=[doc_id]
                )
                self._documents.append({"text": text, "metadata": metadata or {}})
            return True
        except Exception as e:
            logger.error(f"Vector add failed: {e}")
            return False

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar documents."""
        if not self._ready:
            return []
        try:
            if self.backend == "faiss":
                query_embedding = self._embedder.encode([query]).astype("float32")
                distances, indices = self._index.search(query_embedding, min(top_k, len(self._documents)))
                return [
                    {**self._documents[i], "score": float(distances[0][j])}
                    for j, i in enumerate(indices[0])
                    if i < len(self._documents)
                ]
            elif self.backend == "chroma":
                results = self._index.query(
                    query_texts=[query],
                    n_results=min(top_k, self._index.count())
                )
                docs = results.get("documents", [[]])[0]
                metas = results.get("metadatas", [[]])[0]
                return [
                    {"text": doc, "metadata": meta}
                    for doc, meta in zip(docs, metas)
                ]
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
        return []

    @property
    def is_ready(self) -> bool:
        return self._ready