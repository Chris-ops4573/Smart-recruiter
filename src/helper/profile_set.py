from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

#Keyword match scoring
required_skills = {
    # Core retrieval / ranking
    "information retrieval",
    "semantic search",
    "vector search",
    "recommendation systems",
    "learning to rank",
    "bm25",

    # Embeddings
    "embeddings",
    "sentence transformers",

    # Vector databases
    "pinecone",
    "weaviate",
    "qdrant",
    "milvus",
    "faiss",

    # Search infrastructure
    "elasticsearch",
    "opensearch",

    # LLM / matching layer
    "llms",
    "rag",

    # Engineering
    "python"
}

preffered_skills = {
    # Fine tuning
    "lora",
    "qlora",
    "peft",

    # Modern LLM stack
    "langchain",
    "llamaindex",
    "hugging face transformers",

    # Production ML
    "mlops",
    "kubeflow",
    "mlflow",
    "bentoml",

    # Retrieval adjacent
    "pgvector",
    "haystack",

    # General ML
    "machine learning",
    "deep learning",
    "nlp",

    # Infra
    "distributed systems",
    "data pipelines"
}

#JD description for work experience and skils
jd_career_text = """
Built and owned ranking systems.
Built and deployed retrieval systems to production.
Designed candidate matching systems.
Shipped recommendation systems to real users.
Worked on embeddings based retrieval.
Worked on hybrid retrieval and vector search.
Built search infrastructure using vector databases.
Improved search relevance and ranking quality.
Designed evaluation frameworks for ranking systems.
A/B testing.
Offline evaluation.
Online evaluation.
Production machine learning systems.
Production AI systems.
Candidate JD matching systems.
Large scale search systems.
Python engineering.
"""

jd_career_embedding = model.encode(
    jd_career_text,
    normalize_embeddings = True
)

jd_profile_text = """
Senior AI Engineer.
LLMs.
Embeddings.
Semantic Search.
Vector Search.
Information Retrieval.
RAG.
Sentence Transformers.
Vector Databases.
Pinecone.
Weaviate.
Milvus.
Qdrant.
FAISS.
OpenSearch.
Elasticsearch.
LoRA.
QLoRA.
PEFT.
Learning To Rank.
Python.
Machine Learning.
NLP.
Recommendation Systems.
Retrieval Systems.
AI Engineering.
"""

jd_profile_embedding = model.encode(
    jd_profile_text,
    normalize_embeddings = True
)

#Role description for generating reasons
reason_templates = {
    "retrieval and ranking": """
    Designed retrieval systems, search relevance pipelines,
    learning-to-rank models, recommendation systems,
    candidate matching systems, ranking infrastructure,
    offline evaluation and online experimentation.
    """,

    "search": """
    Built search platforms using indexing, BM25, Elasticsearch,
    OpenSearch, query processing, search infrastructure,
    and relevance optimization.
    """,

    "machine learning": """
    Built machine learning systems involving model training,
    feature engineering, prediction systems, evaluation,
    deployment, and production ML workflows.
    """,

    "llm": """
    Built LLM-powered products using prompt engineering,
    fine-tuning, agent workflows, tool use,
    instruction tuning, model adaptation,
    and generative AI applications.
    """,

    "data": """
    Built data pipelines, ETL systems, distributed data
    processing infrastructure, warehouses, streaming systems,
    and analytics platforms.
    """,

    "backend": """
    Built backend services, APIs, databases,
    distributed systems, cloud infrastructure,
    and scalable production software.
    """,

    "mlops": """
    Built ML infrastructure, deployment systems,
    model serving platforms, experiment tracking,
    monitoring, and ML CI/CD pipelines.
    """,

    "platform and tooling": """
    Built developer platforms, infrastructure tooling,
    cloud systems, automation frameworks,
    and internal engineering platforms.
    """
}

reason_names = list(reason_templates.keys())

reason_embeddings = model.encode(
    list(reason_templates.values()),
    normalize_embeddings=True
)