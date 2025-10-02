from dataclasses import dataclass
from utils import (
    EmbeddingFunc,
    logger,
    limit_async_func_call,
    wrap_embedding_func_with_attrs,
)
from typing import Callable
import asyncio

@dataclass
class PathRAG:
    log_level: str = "INFO"

    chunk_token_size: int = 500
    chunk_overlap_token_size: int = 100

    entity_summary_to_max_tokens: int = 500

    embedding_func: EmbeddingFunc
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 16

    llm_model_func: Callable
    llm_model_max_async: int = 10

    @staticmethod
    def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
        """
        Ensure that there is always an event loop available.

        This function tries to get the current event loop. If the current event loop is closed or does not exist,
        it creates a new event loop and sets it as the current event loop.

        Returns:
            asyncio.AbstractEventLoop: The current or newly created event loop.
        """
        try:
            current_loop = asyncio.get_event_loop()
            if current_loop.is_closed():
                raise RuntimeError("Event loop is closed.")
            return current_loop

        except RuntimeError:
            logger.info("Creating a new event loop in main thread.")
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            return new_loop

    def __post_init__(self):
        logger.setLevel(self.log_level)

        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )

        # self.full_docs = self.key_string_value_json_storage_cls(
        #     namespace="full_docs",
        #     global_config=asdict(self),
        #     embedding_func=self.embedding_func,
        # )
        # self.text_chunks = self.key_string_value_json_storage_cls(
        #     namespace="text_chunks",
        #     global_config=asdict(self),
        #     embedding_func=self.embedding_func,
        # )
        # self.chunk_entity_relation_graph = self.graph_storage_cls(
        #     namespace="chunk_entity_relation",
        #     global_config=asdict(self),
        #     embedding_func=self.embedding_func,
        # )


        # self.entities_vdb = self.vector_db_storage_cls(
        #     namespace="entities",
        #     global_config=asdict(self),
        #     embedding_func=self.embedding_func,
        #     meta_fields={"entity_name"},
        # )
        # self.relationships_vdb = self.vector_db_storage_cls(
        #     namespace="relationships",
        #     global_config=asdict(self),
        #     embedding_func=self.embedding_func,
        #     meta_fields={"src_id", "tgt_id"},
        # )
        # self.chunks_vdb = self.vector_db_storage_cls(
        #     namespace="chunks",
        #     global_config=asdict(self),
        #     embedding_func=self.embedding_func,
        # )

    def insert(self, string_or_strings):
        loop = PathRAG.always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(string_or_strings))

    async def ainsert(self, string_or_strings):
        try:
            pass
            # 1. chunking

            # 2. embedding

            # 3. extract entities and relationships

            # 4. create KG

            # 5. update chunks, KG, entities, relationships
        except Exception as e:
            pass
        finally:
            pass

    def query(self, query: str):
        loop = PathRAG.always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query))

    async def aquery(self, query: str):
        response = await kg_query(
            query,
            self.chunk_entity_relation_graph,
            self.entities_vdb,
            self.relationships_vdb,
            self.text_chunks,
        )
        return response