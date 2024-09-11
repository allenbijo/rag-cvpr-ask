from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import pickle
import fetch_papers


def vectorize_papers(papers, paper_names):
	text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
	splits = text_splitter.split_documents(papers)
	if splits:
		embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
		db = FAISS.from_documents(splits, embeddings)
		try:
			dbe = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
			db.merge_from(dbe)
		except:
			pass
		db.save_local("faiss_index")
		try:
			with open("papers.pkl", 'rb') as f:
				existing_data = pickle.load(f)
		except FileNotFoundError:
			existing_data = []

		paper_names.extend(existing_data)

		with open("papers.pkl", 'wb') as f:
			pickle.dump(paper_names, f)


if __name__ == '__main__':
	paper_names = fetch_papers.get_paper_names()
	papers = fetch_papers.fetch_arxiv_papers(paper_names, debug=True)
	vectorize_papers(papers, paper_names)
