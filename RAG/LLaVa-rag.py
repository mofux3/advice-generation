import streamlit as st
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import ServiceContext, VectorStoreIndex, download_loader, set_global_service_context
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts.prompts import SimpleInputPrompt

# Set up LLAVA configuration
name = "llava-hf/llava-v1.6-mistral-7b-hf"
auth_token = ""


# Function to retrieve LLAVA model and processor
@st.cache(allow_output_mutation=True)
def get_model_processor():
    # Create processor
    processor = AutoProcessor.from_pretrained(name, cache_dir='./model/', use_auth_token=auth_token)

    # Create LLAVA model
    model = LlavaForConditionalGeneration.from_pretrained(
        name, torch_dtype=torch.bfloat16, device_map='cuda',
        cache_dir='./workspace', use_auth_token=auth_token
    )
    return processor, model


# Initialize LLAVA components
processor, model = get_model_processor()

# LLAVA system prompt
system_prompt = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as 
helpfully as possible, while being safe. Your answers should not include
any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain 
why instead of answering something not correct. If you don't know the answer 
to a question, please don't share false information.

Your task is to generate an advice report for civil engineers to use in identifying and classifying
pavement defects, such as cracks, potholes and the like. 
You will generate a report that contains at the least the following information:
1. Classification of the type of defect.
2. Current state of the detected defect.
3. Recommended method for solving the issue. Limit this to only the most topical and logical answer.
4. The risks that come when leaving the defect as is.

Your information given will consist of an image, in which the defect will be visible, and a textual description.
If you do not have high confidence in whether or not you detected the defect correctly, respond with an request
to resubmit a different image.<</SYS>>
"""

# LLAVA query wrapper prompt
query_wrapper_prompt = SimpleInputPrompt("{query_str} [/INST]")

# Create LLAVA LLM using the LLAMA index wrapper
llm = HuggingFaceLLM(context_window=4096,
                     max_new_tokens=256,
                     system_prompt=system_prompt,
                     query_wrapper_prompt=query_wrapper_prompt,
                     model=model,
                     tokenizer=processor.tokenizer)

# Create HuggingFace embeddings instance
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create new LLAVA service context instance
service_context = ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embeddings
)

# Set global service context
set_global_service_context(service_context)

# Streamlit app
st.title('LLaVA: Pavement Defect Identification and Classification')

# File uploader for images
uploaded_image = st.file_uploader("Upload an image of the pavement defect. Accepted types: png, jpg, jpeg", type=["png", "jpg", "jpeg"])

# Text input for defect description
prompt = st.text_input('Enter Description of image')


# Function to process image and text using LLAVA
def process_image_and_text(image, prompt):
    if image is not None and prompt:
        image = Image.open(image)

        # Embed image using HuggingFace embeddings
        image_embedding = embeddings.encode_image(image)

        # Run LLAVA retrieval-augmented generation
        query_str = f"{prompt} [INST]"
        response = llm(prompt=query_str, context=image_embedding)

        return response.text


# Display results
if st.button('Generate Report'):
    if uploaded_image is None:
        st.warning("UPload image")
    elif not prompt:
        st.warning("Text description of image in any form. .")
    else:
        response = process_image_and_text(uploaded_image, prompt)
        st.subheader("Generated Report:")
        st.write(response)
