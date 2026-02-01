import mlflow
from mlflow.models import set_model
from databricks.vector_search.client import VectorSearchClient
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

ENDPOINT_NAME = "mydocs"
INDEX_NAME = "my_lab.mydocs_rag.mydocs_chunk_index"
LLM_ENDPOINT = "databricks-gpt-oss-20b"


class RAGPipelineModel(mlflow.pyfunc.PythonModel):
    def __init__(self, endpoint_name=None, index_name=None, llm_endpoint=None):
        self.endpoint_name = endpoint_name or ENDPOINT_NAME
        self.index_name = index_name or INDEX_NAME
        self.llm_endpoint = llm_endpoint or LLM_ENDPOINT
        self.vector_index = VectorSearchClient().get_index(
            endpoint_name=self.endpoint_name, index_name=self.index_name
        )

    def load_context(self, context):
        """
        クライアントの初期化
        """
        self.vsc = VectorSearchClient()
        self.vector_index = self.vsc.get_index(
            endpoint_name=self.endpoint_name,
            index_name=self.index_name
        )
        self.w = WorkspaceClient()

    def query_foundation_model(
        self, prompt: str, endpoint: str = "databricks-gemma-3-12b"
    ) -> str:
        response = self.w.serving_endpoints.query(
            name=endpoint,
            messages=[ChatMessage(role=ChatMessageRole.USER, content=prompt)],
        )
        return response.choices[0].message.content

    @mlflow.trace(name="retrieve_documents", span_type="RETRIEVER")
    def retrieve_documents(self, query: str, num_results: int = 3) -> list:
        documents = self.vector_index.similarity_search(
            query_text=query, columns=["text"], num_results=num_results
        )
        return documents

    @mlflow.trace(name="generate_answer", span_type="LLM")
    def generate_answer(
        self,
        query: str,
        documents: list = None,
    ) -> tuple:
        # Augmentation: 検索結果＋質問をプロンプトとして整形
        prompt = (
            f"以下は参考情報です:\n{documents}\n\n"
            f"質問: {query}\n"
            "参考情報をもとに、わかりやすく日本語で回答してください。"
        )
        # Generation: 基盤モデルAPIで回答生成
        answer = self.query_foundation_model(prompt)
        return answer

    @mlflow.trace(name="rag_pipeline", span_type="CHAIN")
    def rag_pipeline(self, query: str) -> dict:
        # ステップ1: ドキュメント検索
        documents = self.retrieve_documents(query)
        # ステップ2: 回答生成
        answer = self.generate_answer(query=query, documents=documents)
        return {"query": query, "documents": documents, "answer": answer}

    def predict(self, context, model_input):
        query = model_input["query"]
        return self.rag_pipeline(query)


set_model(RAGPipelineModel())
