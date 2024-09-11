import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import ArxivLoader


def get_paper_names(n=10):
	url = f"https://arxiv.org/list/cs.CV/recent?skip=0&show={n}"

	response = requests.get(url)

	soup = BeautifulSoup(response.content, "html.parser")

	papers = []
	for a_tag in soup.find_all('div', class_='list-title mathjax'):
		papers.append(a_tag.text[17:-9])

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