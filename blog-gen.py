import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
import os
from langchain_huggingface.llms import HuggingFaceEndpoint

## Function To get response from LLAma 2 model

def getLLamaresponse(input_text,no_words,blog_style):

    ### LLama2 model
    repo_id = 'mistralai/Mistral-7B-Instruct-v0.2'

    os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'

    HUGGINGFACEHUB_API_TOKEN = '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'

    llm = HuggingFaceEndpoint(
        repo_id=repo_id, temperature=0.5, token=HUGGINGFACEHUB_API_TOKEN,
    )

    template="""
        Write a blog for {blog_style} job profile for a topic {input_text}
        within {no_words} words.
            """

    prompt=PromptTemplate(input_variables=["blog_style","input_text",'no_words'],
                          template=template)

    ## Generate the ressponse from the LLama 2 model
    response=llm(prompt.format(blog_style=blog_style,input_text=input_text,no_words=no_words))
    print(response)
    return response



st.set_page_config(page_title="Generate Blogs",
                    page_icon='🤖',
                    layout='centered',
                    initial_sidebar_state='collapsed')

st.header("Blogsmith 🤖")

input_text=st.text_input("Enter the Blog Topic")

## creating to more columns for additonal 2 fields

col1,col2=st.columns([5,5])

with col1:
    no_words=st.text_input('No of Words')
with col2:
    blog_style=st.selectbox('Writing the blog for',
                            ('Researchers','Data Scientist','Common People'),index=0)

submit=st.button("Generate")

## Final response
if submit:
    st.write(getLLamaresponse(input_text,no_words,blog_style))
