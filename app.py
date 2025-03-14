import streamlit as st
import torch
import sentencepiece as spm

# Load Tokenizer
sp = spm.SentencePieceProcessor()
sp.load("bpe.model")

# Load Model
model = torch.load("transformer_model.pth", map_location=torch.device('cpu'))
model.eval()

def generate_code(input_text):
    """Generate C++ code from pseudocode."""
    input_tokens = sp.encode(input_text, out_type=int)
    input_tensor = torch.tensor([input_tokens])
    with torch.no_grad():
        output_tokens = model(input_tensor)
    output_text = sp.decode(output_tokens.argmax(dim=-1).squeeze().tolist())
    return output_text

# Streamlit UI
st.title("Pseudocode to C++ Code Generator")
st.write("Enter your pseudocode and get the corresponding C++ code.")

user_input = st.text_area("Enter Pseudocode:")
if st.button("Generate Code"):
    if user_input.strip():
        output_code = generate_code(user_input)
        st.code(output_code, language='cpp')
    else:
        st.warning("Please enter some pseudocode!")
