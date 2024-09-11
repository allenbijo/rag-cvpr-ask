import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import ArxivLoader
import pickle


# Get paper names from arxiv
def get_paper_names():
	# Get 10 papers after k
	def find_papers(k=0):
		url = f"https://arxiv.org/list/cs.CV/recent?skip={k}&show=10"

		response = requests.get(url)

		soup = BeautifulSoup(response.content, "html.parser")

		papers = []
		for a_tag in soup.find_all('div', class_='list-title mathjax'):
			papers.append(a_tag.text[17:-9])
		return papers

	# Check if papers have already been fetched
	try:
		with open('papers.pkl', 'rb') as f:
			papers = []
			k = 0
			existing_data = pickle.load(f)
			while (paper := find_papers(k))[-1] not in existing_data:
				k += 10
				papers.extend(paper)
			else:
				for i in range(paper.index(existing_data[0])):
					papers.append(paper[i])
	except:
		papers = find_papers(15)

	return papers


def fetch_arxiv_papers(titles, debug=False):
	all_docs = []
	for title in titles:
		try:
			loader = ArxivLoader(
				query=f'title={title}',  # Search in title
				load_max_docs=1,  # Limit to 1 doc per title
				load_all_available_meta=True  # Get all available metadata
			)
			docs = loader.load()
			if docs:
				all_docs.extend(docs)
				if debug: print(f"Retrieved paper: {title}")
			else:
				if debug: print(f"No paper found for title: {title}")
		except Exception as e:
			print(f"Error retrieving paper '{title}': {str(e)}")
	return all_docs


if __name__ == '__main__':
	paper_names = get_paper_names(10)
	papers = fetch_arxiv_papers(paper_names)
	for doc in papers:
		print(f"\nTitle: {doc.metadata['Title']}")
		print(f"Authors: {doc.metadata['Authors']}")
		print(f"Published: {doc.metadata['Published']}")