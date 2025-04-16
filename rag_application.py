import os
import lazyllm
from lazyllm.tools.rag import Document
from lazyllm import SentenceSplitter
from lazyllm import pipeline, parallel, Retriever, Reranker, bind

def build_ppl():
    prompt = ('You will play the role of an AI Q&A assistant and complete a dialogue task. '
              'In this task, you need to provide your answer based on the given context and question.')

    documents = Document(dataset_path=os.path.join(os.getcwd(), "rag_data"),
                         embed=lazyllm.OnlineEmbeddingModule(source="glm", embed_model_name="embedding-2"), manager=False)
    documents.create_node_group(name="sentences", transform=SentenceSplitter, chunk_size=1024, chunk_overlap=100)

    with pipeline() as ppl:
        with parallel().sum as ppl.prl:
            ppl.prl.retriever1 = Retriever(documents, group_name="sentences", similarity="cosine",
                                           similarity_cut_off=0.003, topk=3)
            ppl.prl.retriever2 = Retriever(documents, "CoarseChunk", "bm25_chinese", 0.003, topk=3)
        ppl.reranker = Reranker("ModuleReranker",
                                model=lazyllm.OnlineEmbeddingModule(type="rerank", source="glm",
                                                                    embed_model_name="rerank"),
                                topk=1, output_format='content', join=True) | bind(query=ppl.input)
        ppl.formatter = (lambda nodes, query: dict(context_str=nodes, query=query)) | bind(query=ppl.input)
        ppl.llm = lazyllm.OnlineChatModule(source='glm',
                                           model="glm-4",
                                           stream=False).prompt(lazyllm.ChatPrompter(prompt, extra_keys=["context_str"]))

    return ppl

if __name__ == "__main__":
    ppl = build_ppl()
    while True:
        query = input("请输入你要问的问题，退出请输入(quit or q):")
        if query.lower() in ["quit", 'q']:
            break
        ret = ppl(query)
        print(f"response: {ret}")
