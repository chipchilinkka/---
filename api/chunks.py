from dotenv import load_dotenv
import os
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS

load_dotenv('.env')

class Chunk():
    def __init__(self):
        self.base_load()
        
    #Метод загрузки бз
    def base_load(self):
        with open('base/aero.txt', 'r', encoding='utf-8') as file:
            document = file.read()
            
        #создаем список чанков
        source_chunks = []
        splitter = CharacterTextSplitter(separator=' ', chunk_size = 1000)
        
        for chunk in splitter.split_text(document):
            source_chunks.append(Document(page_content= chunk, metadata = {}))
            
        #создаем индекусную базу
        embeddings = OpenAIEmbeddings()
        self.db = FAISS.from_documents(source_chunks, embeddings)
        
        #инструкция
        self.system = '''
            Ты-консультант по страхованию ответственности аэропортов и авиационных товаропроизводителей, ответь на вопрос клиента на основе документа с информацией.
            Не придумывай ничего от себя, отвечай по документу, не упоминай документ с информацией для ответа клиенту
        '''
            
    #метод запроса
    def get_answer(self, query: str):
        #получаем релевантные отрезки из бз
        docs = self.db.similarity_search(query, k=4)
        message_content = '\n'.join([f'{doc.page_content}' for doc in docs])
        
        #инструкция
        user = f'''
            Ответь на вопрос клиента. Не упоминай документ с информацией в ответе.
            Документ с информацией клиенту: {message_content}\n\n
            Вопрос клиента: \n{query}
        '''
        
        messages = [
            {'role': 'system', 'content': self.system},
            {'role': 'user', 'content': user}
        ]
        
        #обращение
        client = OpenAI()
        response = client.chat.completions.create(
            model = 'gpt-4o-mini',
            messages = messages,
            temperature = 0
        )
        
        return response.choices[0].message.content