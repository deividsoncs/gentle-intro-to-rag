# Construindo um aplicação de RAG (Retrieval-Augmented Generation - Reração Aumentada de Recuperação) 

###  Introdução ao RAG: 

A RAG é uma técnica de inteligência artificial relativamente nova que pode melhorar a qualidade da IA generativa, 
permitindo que modelos de linguagem grandes (LLMs) aproveitem recursos de dados adicionais [sem retreinamento].

Os modelos de RAG [constroem repositórios de conhecimento] com base nos dados da [própria organização], e estes repositórios
 podem ser atualizados continuamente para ajudar a IA generativa a fornecer respostas contextuais e oportunas.

A implementação da RAG requer tecnologias como [bancos de dados vetoriais], que permitem a codificação rápida de novos dados, 
e pesquisas nesses dados para alimentar o LLM.

Referências: 
Introdução ao RAG [https://github.com/svpino/gentle-intro-to-rag]
O que é uma RAG:  https://www.oracle.com/br/artificial-intelligence/generative-ai/retrieval-augmented-generation-rag/
O que é Embeddings: [https://www.cloudflare.com/pt-br/learning/ai/what-are-embeddings]


### Fluxo criacional da base:

Document PDF 
	-> PDF LOADER 
		-> PAGES 
			-> SPLITTER WITH Overlapping chunk of characters
				-> Embeddings (Cria palavras chaves dentro dos chunks, que dão indicativo que os chunks pode fazer parte do mesmo contexto)
					-> StoreVector(ex.:FAISS)(Banco de dados de vetores que aramzenará os os chunks e provera a comparação em um plano multidimenssional)  
						
### Fluxo de Retrival Information:

Question (Retriver fara o mesmo processo que fez para salvar as informações porém agora para procurar a similaridade)
	-> | Retriever : embedding model -> Retrieve similar chunks | 
		-> Vector Store (FAISS, POSTGRE)
			-> Similar chunks to analise.

### Configurando o Modelo:
Usaremos o modelo Ollama local, após criar o modelo, nós podemos invocá-lo com o questionamento para obter a resposta:

	Question -> Model -> Response
	
```python
from langchain_ollama import ChatOllama
model = ChatOllama(model=MODEL , temperature=0) # 0 - less creative 1 - more creative
print(chain.invoke("Qual a capital de Israel?")

```python

### Fazendo o Parsing da resposta do modelo 
O Lanchain nos possibiltas com a instrução | (pipe) pegar uma saáida de um comando e inserir na sequencia no input de outro, que por sua vez fará o "parse" 
para a manipulação assim desejada, possibilidando assim uma corrente de parses (ou seja a "language" "chain").

	question -> CHAIN: [ model --response--> perser ] -> answer 

```python
from langchain_core import StrOutputParser
parser = StrOutputParser()
chain = model | parser
print(chain.invoke("Quem é o protagonista da série the office?"))
```python

### Configurando o prompt

Agora iremos submeter o contexto do PDF, ora armazena no FAISS (nosso Storage Vector), juntamente com a questão a ser feita no nosso **prompt template**
criando assim a entrada para nosso modelo.

```python
from langchain.prompts import PromptTemplate

template = """
Você é um assistente que fornece respostas para perguntas com base em um contexto dado.

Responda à pergunta com base no contexto. Se não puder responder à pergunta, responda "Eu não sei".

Seja o mais conciso possível e vá direto ao ponto.

Contexto: {contexto}

Pergunta: {pergunta}
"""

prompt = PromptTemplate.from_template(template)
print(prompt.format(context="Here is some context", question="Here is a question"))
```python


### Agora Adicionaremos o prompt a nossa CHAIN
  
  Context |              *chain*
		  --->[ PROMPT -> MODEL -> PARSER ] --> ANSWER
  Question|
  
```python

chain = prompt | model | parser

chain.invoke({
    "context": {}, 
    "question": {}
})

```python

========================
# Uma opinião do Reddit: https://www.reddit.com/r/LangChain/comments/18xp9xi/rag_for_pdf_with_tables/?tl=pt-br
Embora este seja um post antigo do Reddit, estou comentando porque a solução para este problema continua a evoluir à medida que os 
LLMs melhoram e a pilha de tecnologia de IA pronta para produção se estabiliza.

A solução requer dois fluxos de trabalho: 
1. Um extrator de texto genérico e agnóstico a documentos para preparar os documentos para
   o consumo dos LLMs. Por exemplo:
	- LLMWhisperer (Manipula bem tabelas com várias linhas): https://unstract.com/llmwhisperer/ 
	- Unstructred.io: http://unstructred.io/

2. Um sistema para analisar o resultado do extrator de texto, alimentá-lo aos LLMs e entregar os resultados em formato estruturado (JSON).
   Por exemplo, Lanchain + Pydantic, ou Unstract.

Alguns exemplos para começar:

https://github.com/Zipstack/structured-extraction

https://github.com/Zipstack/llmwhisperer-table-extraction 