from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.node_parser import SimpleNodeParser
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from loguru import logger
import os

load_dotenv()
qdrant_host = os.getenv("QDRANT_HOST")
qdrant_port = os.getenv("QDRANT_PORT")
qdrant_collection = os.getenv("QDRANT_COLLECTION")

logger.debug(f"QDRANT_HOST: {qdrant_host}")
logger.debug(f"QDRANT_PORT: {qdrant_port}")
logger.debug(f"QDRANT_COLLECTION: {qdrant_collection}")

try:
    qdrant_port = int(qdrant_port)
except (TypeError, ValueError):
    logger.error("QDRANT_PORT is not set correctly in the environment variables.")
    raise

if not qdrant_collection:
    logger.error("QDRANT_COLLECTION is not set in the environment variables.")
    raise ValueError("QDRANT_COLLECTION is required but not set.")

client = QdrantClient(host=qdrant_host, port=qdrant_port)
vector_store = QdrantVectorStore(client=client, collection_name=qdrant_collection)


def create_or_update_index(documents):
    if not documents:
        return None
    
    # Filter out documents with empty or invalid text content
    filtered_docs = []
    for doc in documents:
        try:
            text = doc.get_content() if hasattr(doc, 'get_content') else (doc.text if hasattr(doc, 'text') else str(doc))
            # Ensure text is valid string and not empty
            # Note: doc.text is read-only, so we can't modify it - just validate and pass through
            if text and isinstance(text, str) and text.strip() and len(text.strip()) > 0:
                # Document has valid text, include it
                filtered_docs.append(doc)
        except Exception as e:
            # Log but skip problematic documents
            logger.warning(f"Skipping document due to error: {e}")
            continue
    
    if not filtered_docs:
        logger.warning("No valid documents to index after filtering")
        return None
    
    # Create node parser
    node_parser = SimpleNodeParser.from_defaults(
        chunk_size=1024,
        chunk_overlap=20,
        include_metadata=True
    )
    
    # Parse documents into nodes first
    all_nodes = []
    for doc in filtered_docs:
        try:
            nodes = node_parser.get_nodes_from_documents([doc])
            # Filter out nodes with empty or invalid text
            for node in nodes:
                try:
                    node_text = node.get_content() if hasattr(node, 'get_content') else (node.text if hasattr(node, 'text') else str(node))
                    # Ensure node text is valid string and not empty
                    if node_text and isinstance(node_text, str) and node_text.strip() and len(node_text.strip()) > 0:
                        all_nodes.append(node)
                    else:
                        logger.debug(f"Skipping node with empty/invalid text")
                except Exception as e:
                    logger.debug(f"Skipping node due to error: {e}")
                    continue
        except Exception as e:
            logger.warning(f"Error parsing document into nodes: {e}")
            continue
    
    if not all_nodes:
        logger.warning("No valid nodes to index after filtering")
        return None
    
    logger.info(f"Created {len(all_nodes)} valid nodes from {len(filtered_docs)} documents")
    
    # Pre-embed nodes individually to filter out any that fail embedding
    embed_model = Settings.embed_model
    valid_embedded_nodes = []
    failed_count = 0
    
    for node in all_nodes:
        try:
            node_text = node.get_content() if hasattr(node, 'get_content') else (node.text if hasattr(node, 'text') else str(node))
            
            # Ensure text is a valid string
            if not isinstance(node_text, str):
                node_text = str(node_text) if node_text is not None else ""
            
            # Clean text
            node_text = node_text.replace('\x00', '').replace('\ufffd', '')
            node_text = ' '.join(node_text.split())
            
            # Skip if empty after cleaning
            if not node_text or len(node_text.strip()) < 10:
                failed_count += 1
                continue
            
            # Try to generate embedding
            try:
                embedding = embed_model.get_text_embedding(node_text)
                if embedding and len(embedding) > 0:
                    # Set embedding on node
                    node.embedding = embedding
                    valid_embedded_nodes.append(node)
                else:
                    failed_count += 1
                    logger.debug(f"Node embedding generation returned empty")
            except Exception as embed_error:
                failed_count += 1
                logger.debug(f"Node embedding failed: {embed_error}")
                continue
                
        except Exception as e:
            failed_count += 1
            logger.debug(f"Error processing node: {e}")
            continue
    
    if failed_count > 0:
        logger.info(f"Filtered out {failed_count} nodes that failed embedding")
    
    if not valid_embedded_nodes:
        logger.warning("No nodes with valid embeddings after processing")
        return None
    
    logger.info(f"Successfully embedded {len(valid_embedded_nodes)} nodes")
    
    # Create index from pre-embedded nodes
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(
        nodes=valid_embedded_nodes,
        storage_context=storage_context
    )
    return index
