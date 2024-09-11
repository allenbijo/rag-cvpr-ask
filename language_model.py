from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import HumanMessagePromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def get_data():
	try:
		embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
		dbe = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
		return dbe

	except:
		return False


def prepare_model():
	model_name = "google/gemma-2-2b-it"
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModelForCausalLM.from_pretrained(model_name)

	generation_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=2000, device=0)
	llm = HuggingFacePipeline(pipeline=generation_pipeline)
	return llm


def setup(llm):
	if db := get_data():
		retriever = db.as_retriever(search_type="similarity")
	else:
		return "No data available"

	prompt = ChatPromptTemplate(input_variables=['context', 'question'], messages=[HumanMessagePromptTemplate(
		prompt=PromptTemplate(input_variables=['context', 'question'],
		                      template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.\nQuestion: {question} \nContext: {context} \nAnswer:"))])

	def format_docs(docs):
		out = ""
		for i, doc in enumerate(docs):
			out += f"Document{i} Title: {doc.metadata['Title']}, Authors: {doc.metadata['Authors']}, Published: {doc.metadata['Published']}"
			out += doc.page_content
			out += '\n\n'
		return out

	class CleanOutputParser(StrOutputParser):
		def parse(self, output: str) -> str:
			# Strip out everything except the answer part
			if "Answer:" in output:
				answer = output.split("Answer:")[-1].strip()
			else:
				answer = output
			return answer

	rag_chain = (
			{"context": retriever | format_docs, "question": RunnablePassthrough()}
			| prompt
			| llm
			| CleanOutputParser()
	)
	return rag_chain


if __name__ == '__main__':
	llm = prepare_model()
	chain = setup(llm)
	# chain.invoke('Who wrote the paper about LEIA')