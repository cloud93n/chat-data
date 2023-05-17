from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
import os

load_dotenv()

persist_directory = os.environ.get('PERSIST_DIRECTORY')


from constants import CHROMA_SETTINGS

def main():
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base")
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    callbacks = [StreamingStdOutCallbackHandler()]
   
    qa = RetrievalQA.from_chain_type(llm=OpenAI(callbacks=callbacks), chain_type="stuff", retriever=retriever, return_source_documents=True)
    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        
        # Get the answer from the chain
        res = qa(query)    
        answer, docs = res['result'], res['source_documents']

        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)
        
        # Print the relevant sources used for the answer
        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)

if __name__ == "__main__":
    main()