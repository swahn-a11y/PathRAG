import os
from PathRAG import PathRAG, QueryParam
from PathRAG.llm import gpt_4o_mini_complete

WORKING_DIR = "./PATHRAG_CACHE"

# os.environ["OPENAI_API_KEY"] = api_key
# base_url="https://api.openai.com/v1"
# os.environ["OPENAI_API_BASE"]=base_url


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = PathRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete,  
)


data_file="./a.txt"
question="What is the capital of France?"
with open(data_file) as f:
    rag.insert(f.read())

# print(rag.query(question, param=QueryParam(mode="hybrid")))

# if __name__ == "__main__":

#     import asyncio
#     import random

#     def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
#         """
#         Ensure that there is always an event loop available.

#         This function tries to get the current event loop. If the current event loop is closed or does not exist,
#         it creates a new event loop and sets it as the current event loop.

#         Returns:
#             asyncio.AbstractEventLoop: The current or newly created event loop.
#         """
#         try:

#             current_loop = asyncio.get_event_loop()
#             if current_loop.is_closed():
#                 raise RuntimeError("Event loop is closed.")
#             return current_loop

#         except RuntimeError:

#             logger.info("Creating a new event loop in main thread.")
#             new_loop = asyncio.new_event_loop()
#             asyncio.set_event_loop(new_loop)
#             return new_loop

#     # print(asyncio.get_event_loop())


#     async def fetch_one(i):
#         print(i)
#         await asyncio.sleep(1)   # I/O 대기 시 '양보'
#         return f"done {i}"

#     # async def main():
#     #     # 단순 순차
#     #     a = await fetch_one(1)
#     #     b = await fetch_one(2)
#     #     print(a, b)

#     # asyncio.run(main())  # 진입점

#     # async def main():
#     #     t1 = asyncio.create_task(fetch_one(1))
#     #     t2 = asyncio.create_task(fetch_one(2))
#     #     print(await t1, await t2)


#     #     # 또는 한번에
#     #     results = await asyncio.gather(fetch_one(3), fetch_one(4))
#     #     print(results)

#     async def producer(q):
#         for i in range(20):
#             await q.put(i)
#         await q.put(None)  # 종료 신호

#     async def consumer(q):
#         while True:
#             item = await q.get()
#             if item is None:
#                 q.task_done()
#                 break
#             await asyncio.sleep(random.random())
#             q.task_done()

#     async def main():
#         q = asyncio.Queue()
#         await asyncio.gather(producer(q), consumer(q))
#         await q.join()

#     asyncio.run(main())  # 진입점









