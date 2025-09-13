import os  
from langchain_astradb import AstraDBVectorStore
from typing import List
from langchain_core.documents import Document
from dotenv import load_dotenv
from utils.model_loader import ModelLoader
from utils.config_loader import load_config


class Retriever:
    def __init__(self):
        """
        summary
        """
        self.model_loader=ModelLoader()
        self._load_env_variables()
        self.config=load_config()
        self.vstore = None
        self.retriever = None

    def _load_env_variables(self):
        """
        summary
        """
        load_dotenv()
        required_vars = ["GOOGLE_API_KEY", "ASTRA_DB_API_ENDPOINT", "ASTRA_DB_APPLICATION_TOKEN", "ASTRA_DB_KEYSPACE"]
        missing_vars = [var for var in required_vars if os.getenv(var) is None]

        if missing_vars:
            raise EnvironmentError(f"Missing environment variables: {missing_vars}")
        
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.db_api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
        self.db_application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        self.db_keyspace = os.getenv("ASTRA_DB_KEYSPACE")
        
    def load_retriever(self):
        """
        summary
        """
        if not self.vstore:
            collection_name = self.config["astra_db"]["collection_name"]

            self.vstore = AstraDBVectorStore(
                embedding = self.model_loader.load_embeddings(),
                collection_name=collection_name,
                api_endpoint=self.db_api_endpoint,
                token=self.db_application_token,
                namespace=self.db_keyspace

            )      

        if not self.retriever:
            top_k = self.config["retriever"]["top_k"] if "retriever" in self.config else 3
            retriever = self.vstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})  
            print("Retriever loaded successfully")
            return retriever

    def call_retriever(self,user_query):
        """
        summary
        """
        retriever = self.load_retriever()
        output = retriever.invoke(user_query)
        return output

if __name__ == "__main__":
    retriever_obj=Retriever()
    user_query = "Can you suggest good budget laptops for students?"
    result = retriever_obj.call_retriever(user_query)

    for idx,doc in enumerate(result):
        print(f"Result {idx} : {doc.page_content}\n Metadata: {doc.metadata}\n")