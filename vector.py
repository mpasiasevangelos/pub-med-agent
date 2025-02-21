import os
import warnings
from dotenv import load_dotenv
import xml.etree.ElementTree as ET
from langchain_ollama import OllamaEmbeddings
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

# Suppress warnings and load environment variables
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore")
load_dotenv()

# Function to parse XML and extract titles and abstracts
def parse_pubmed_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    documents = []
    for article in root.findall(".//PubmedArticle"):
        title_element = article.find(".//ArticleTitle")
        abstract_element = article.find(".//AbstractText")
        
        # Check if title and abstract exist
        title = title_element.text if title_element is not None else None
        abstract = abstract_element.text if abstract_element is not None else None
        
        # Only add the document if both title and abstract are present
        if title and abstract:
            documents.append({"title": title, "abstract": abstract})
    return documents


# Directory containing your XML files
xml_directory = "./"

# Collect all XML files
xml_files = [os.path.join(xml_directory, file) for file in os.listdir(xml_directory) if file.endswith(".xml")]

print(xml_files)

# Extract titles and abstracts from all XML files
all_docs = []
for xml_file in xml_files:
    docs = parse_pubmed_xml(xml_file)
    all_docs.extend(docs)
print(f"Extracted {len(all_docs)} documents from {len(xml_files)} XML files.")

# Initialize Ollama embeddings
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url='http://localhost:11434'
)

# Embed titles and abstracts
embedded_docs = []
for doc in all_docs:
    text = f"{doc['title']} {doc['abstract']}"
    vector = embeddings.embed_query(text)
    embedded_docs.append({"text": text, "vector": vector})

# Initialize FAISS vector store
dimension = len(embedded_docs[0]["vector"])
index = faiss.IndexFlatIP(dimension)
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

from langchain.schema import Document

# Add documents to the vector store in batches
batch_size = 100  # Adjust batch size as needed
for i in range(0, len(embedded_docs), batch_size):
    batch = embedded_docs[i:i + batch_size]
    
    # Convert each document to a Document object
    documents = [Document(page_content=doc["text"]) for doc in batch]
    
    # Add the documents to the vector store
    vector_store.add_documents(documents)

# Save the vector store locally
db_name = "pubmed_titles_abstracts"
vector_store.save_local(db_name)
print(f"Vector store saved as '{db_name}'.")

# Test the vector store
question = "What chemicals are associated with cancer?"
results = vector_store.search(query=question, k=5, search_type='similarity')
print("Search results:")
for result in results:
    print(result)
