from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import fetch_papers

def vectorize_papers(papers):
	text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
	splits = text_splitter.split_documents(papers)
	db = FAISS.from_documents(splits, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
	db.save_local("faiss_index")


if __name__ == '__main__':
	paper_names = fetch_papers.get_paper_names(10)
	papers = fetch_papers.fetch_arxiv_papers(paper_names, debug=True)
	vectorize_papers(papers)
