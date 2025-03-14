import streamlit as st
import torch
import sentencepiece as spm
import torch.nn as nn

# Define Model Architecture (Modify this according to your model)
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, 256)  
        self.fc = nn.Linear(256, output_dim)  

    def forward(self, x):
        x = self.embedding(x)
        x = self.fc(x)
        return x

# Load Tokenizer
sp = spm.SentencePieceProcessor()
sp.load("bpe.model")

# Initialize Model (Modify input_dim and output_dim as per your model)
input_dim = 5000  # Example vocabulary size
output_dim = 5000  # Example output size
model = TransformerModel(input_dim, output_dim)

# Load State Dict (instead of torch.load)
model.load_state_dict(torch.load("transformer_model.pth", map_location=torch.device('cpu')))
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
