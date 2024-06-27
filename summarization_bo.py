from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import streamlit as st


def model(tokens=200):
  model_id = "facebook/bart-large-cnn"

  tokenizer = AutoTokenizer.from_pretrained(model_id)

  model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

  pipe = pipeline("summarization", model=model, tokenizer=tokenizer, max_new_tokens=tokens)

  hf = HuggingFacePipeline(pipeline=pipe)

  template = """Summarize the given text
  {t}
  """

  prompt = PromptTemplate.from_template(template)

  chain = prompt | hf

  return chain

def main():
  # st.set_page_config(page_title='🦜🔗 Text Summarization App')
  st.title('🦜🔗 Text Summarization App')
  # with st.chat_message("user"):
  #     st.write("Hello 👋")


  # tokens = st.selectbox(
  #   "Select summary length",
  #   (100, 200, 500))

  chain=model(tokens)

  # Text inpu
  txt_input = st.text_area('Enter your text', '', height=200)
  result = []
  with st.form('summarize_form', clear_on_submit=True):
      submitted = st.form_submit_button('Submit')
      #if submitted and openai_api_key.startswith('sk-'):
      if submitted:
          with st.spinner('Generating summary...'):
            # st.write("submitted please wait ")
            t = txt_input
            st.write(chain.invoke({"t": t}))

              # docs = chunks_and_document(txt_input)
              # response = chains_and_response(docs)
              # result.append(response)

if __name__=='__main__':
  main()
