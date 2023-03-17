import streamlit as st
from transformers import BioGptTokenizer, BioGptForCausalLM
import torch
@st.cache_resource
def init():
	model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
	tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
	return model, tokenizer
model, tokenizer = init()
if 'response' not in st.session_state:
	st.session_state['response'] = []
st.title("BioGPT demo!")
min_length = st.slider("Select the minimum length", 10, 100)
max_length = st.slider("Select the Maximum length", 100, 300)
question = st.text_input("What's your prompt?")
col1, _, _, _, _, _, col2 = st.columns(7)
with col1:
	if (st.button("Submit")):
		inputs = tokenizer(question, return_tensors='pt')
		with torch.no_grad():
			beam_output = model.generate(**inputs, min_length=min_length, max_length=max_length, num_beams=5)
			ans = tokenizer.decode(beam_output[0], skip_special_tokens=True)
		st.session_state['response'].append(ans)
with col2:
	if (st.button("Clear")):
		st.session_state['response'] = []

for r in st.session_state['response']:
	st.write(r)