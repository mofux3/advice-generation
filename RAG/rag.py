# Import streamlit for app dev
# アプリ開発用にstreamlitをインポート
import streamlit as st

# Import transformer classes for generation
# 生成用のtransformerクラスをインポート
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
# Import torch for datatype attributes 
# データ型属性用にtorchをインポート
import torch
# Import the prompt wrapper...but for llama index
# プロンプトラッパーをインポート...ただしllama index用
from llama_index.prompts.prompts import SimpleInputPrompt
# Import the llama index HF Wrapper
# llama index HFラッパーをインポート
from llama_index.llms import HuggingFaceLLM
# Bring in embeddings wrapper
# 埋め込みラッパーを持ち込む
from llama_index.embeddings import LangchainEmbedding
# Bring in HF embeddings - need these to represent document chunks
# HF埋め込みを持ち込む - ドキュメントチャンクを表現するために必要
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# Bring in stuff to change service context
# サービスコンテキストを変更するためのものを持ち込む
from llama_index import set_global_service_context
from llama_index import ServiceContext
# Import deps to load documents 
# ドキュメントをロードするための依存関係をインポート
from llama_index import VectorStoreIndex, download_loader
from pathlib import Path

# Define variable to hold llama2 weights naming 
# llama2の重み付け名を保持する変数を定義
name = "meta-llama/Llama-2-7b-chat-hf"
# Set auth token variable from hugging face 
# Hugging Faceから認証トークン変数を設定
auth_token = "YOUR_TOKEN"

@st.cache_resource
def get_tokenizer_model():
    # Create tokenizer
    # トークナイザーを作成
    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir='./model/', use_auth_token=auth_token)

    # Create model
    # モデルを作成
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, device_map='cuda',
        cache_dir='./workspace', token=auth_token
    )
    return tokenizer, model

tokenizer, model = get_tokenizer_model()

# Create a system prompt 
# システムプロンプトを作成
system_prompt = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as 
helpfully as possible, while being safe. Your answers should not include
any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain 
why instead of answering something not correct. If you don't know the answer 
to a question, please don't share false information.

Your goal is to provide synthetic datasets for the purpose of finetuning LLM models.<</SYS>>
"""
# Throw together the query wrapper
# クエリラッパーをまとめる
query_wrapper_prompt = SimpleInputPrompt("{query_str} [/INST]")

# Create a HF LLM using the llama index wrapper 
# llama indexラッパーを使用してHF LLMを作成
llm = HuggingFaceLLM(context_window=4096,
                     max_new_tokens=256,
                     system_prompt=system_prompt,
                     query_wrapper_prompt=query_wrapper_prompt,
                     model=model,
                     tokenizer=tokenizer)

# Create and dl embeddings instance  
# 埋め込みインスタンスを作成してダウンロード
embeddings = LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)

# Create new service context instance
# 新しいサービスコンテキストインスタンスを作成
service_context = ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embeddings
)
# Set service context
# サービスコンテキストを設定
set_global_service_context(service_context)

# Download PDF Loader 
# PDFローダーをダウンロード
PyMuPDFReader = download_loader("PyMuPDFReader")
# Create PDF Loader
# PDFローダーを作成
loader = PyMuPDFReader()
# Load documents 
# ドキュメントをロード
documents = loader.load(file_path=Path('Link to your document/documents'), metadata=True)

# Create an index
# インデックスを作成
index = VectorStoreIndex.from_documents(documents)
# I don't get this at all
# これは全く理解できません
query_engine = index.as_query_engine()

# Create centered main title 
# 中央に配置されたメインタイトルを作成
st.title('<<Title here>>')
# Create a text input box for the user
# ユーザー用のテキスト入力ボックスを作成
prompt = st.text_input('Input prompt')

# If the user hits enter
# ユーザーがEnterキーを押した場合
if prompt:
    response = query_engine.query(prompt)
    # ...and write it out to the screen
    # ...そしてそれを画面に書き出す
    st.write(response)

    # Display raw response object
    # 生の応答オブジェクトを表示
    with st.expander('Response Object'):
        st.write(response)
    # Display source text
    # ソーステキストを表示
    with st.expander('Source Text'):
        st.write(response.get_formatted_sources())
