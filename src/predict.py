import mlflow
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()
vector_index = vsc.get_index(endpoint_name="mydocs", index_name="my_lab.mydocs_rag.mydocs_chunk_index")

# WorkspaceClientの初期化
w = WorkspaceClient()

def call_foundation_model_api(prompt: str, endpoint: str = "databricks-gpt-oss-20b") -> str:
    """
    Databricks Foundation Model APIを呼び出す

    Args:
        prompt: LLMへの入力プロンプト
        endpoint: Foundation Modelのエンドポイント名

    Returns:
        LLMからの応答テキスト
    """
    response = w.serving_endpoints.query(
        name=endpoint,
        messages=[
            ChatMessage(role=ChatMessageRole.USER, content=prompt)
        ]
    )
    return response.choices[0].message.content

def build_prompt(query, docs_texts):
    context = "\n\n".join(docs_texts)
    return f"参考情報:\n{context}\n\n質問: {query}\n日本語でわかりやすく答えてください。"

def call_llm(prompt, max_tokens=512):
    return call_foundation_model_api(prompt)

def predict(input: dict) -> dict:
    query = input["query"]
    docs = vector_index.similarity_search(query_text=query, columns=["text"], num_results=5)
    docs_texts = [d["text"] for d in docs]
    prompt = build_prompt(query, docs_texts)
    answer = call_llm(prompt)
    return {"answer": answer, "sources": docs}
