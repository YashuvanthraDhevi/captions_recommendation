import streamlit as st
import torch
from PIL import Image
import io
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

st.set_page_config(page_title="Caption Recommendation Tool", layout="wide")

@st.cache_resource
def load_clip_model():
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

def load_and_preprocess_image(image):
    image = image.convert("RGB")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    inputs = processor(images=image, return_tensors="pt")
    return inputs, processor

def generate_image_embeddings(inputs, model):
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return image_features

def match_captions(image_features, captions, clip_model, processor):
    # Get text embeddings for the captions
    text_inputs = processor(text=captions, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**text_inputs)

    # Calculate cosine similarity between image and text features
    image_features = image_features.detach().cpu().numpy()
    text_features = text_features.detach().cpu().numpy()

    similarities = cosine_similarity(image_features, text_features)

    # Find the best matching captions
    best_indices = similarities.argsort(axis=1)[0][::-1]  
    best_captions = [captions[i] for i in best_indices]

    return best_captions, similarities[0][best_indices].tolist()

def image_captioning(image, candidate_captions, model, processor):  
    inputs, _ = load_and_preprocess_image(image)
    image_features = generate_image_embeddings(inputs, model)

    best_captions, similarities = match_captions(image_features, candidate_captions, model, processor)
    return best_captions, similarities

# App layout
st.title("📸 Caption Recommendation Tool")
st.markdown("""
This app uses CLIP (Contrastive Language-Image Pre-training) to find the best matching captions for your images.
Upload an image and get the top caption recommendations!
""")

# Load the model on first run
with st.spinner("Loading CLIP model... This may take a moment."):
    model, processor = load_clip_model()

# Predefined path for the Excel file with captions (assuming it's stored on the server)
CAPTION_FILE_PATH = "social_media_captions.xlsx"

# Load captions from the Excel file on the backend
try:
    df = pd.read_excel(CAPTION_FILE_PATH)
    captions = df.iloc[:, 0].dropna().tolist()  # Assumes captions are in the first column
    st.success("Captions loaded successfully from backend!")
except Exception as e:
    st.error(f"Error loading captions: {e}")
    captions = []

# Main section for image upload
st.header("Upload Your Image")
uploaded_image_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_image_file is not None and captions:
    # Display the uploaded image
    col1, col2 = st.columns([1, 1])
    with col1:
        image = Image.open(uploaded_image_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Process the image and get caption recommendations
    with st.spinner("Analyzing image..."):
        try:
            best_captions, similarities = image_captioning(image, captions, model, processor)
            
            # Display results
            with col2:
                st.subheader("Top Caption Recommendations")
                
                # Get the top results
                top_best_captions = best_captions[:5]
                top_similarities = similarities[:5]
                
                # Display each caption with a side-by-side similarity bar
                for i, (caption, similarity) in enumerate(zip(top_best_captions, top_similarities)):
                    score_percentage = round(similarity * 100, 2)
                    
                    # Create columns for the caption and progress bar
                    cap_col, bar_col = st.columns([2, 3])
                    
                    with cap_col:
                        st.markdown(f"**{i+1}. {caption}**")
                    
                    with bar_col:
                        st.progress(score_percentage/100)
                        st.markdown(f"{score_percentage}%")
                    
                    st.divider()
                
                # Option to download best caption
                st.download_button(
                    label="Download Best Caption",
                    data=top_best_captions[0],
                    file_name="best_caption.txt",
                    mime="text/plain"
                )
                
        except Exception as e:
            st.error(f"Error processing image: {e}")
else:
    # Show a placeholder when no image or captions are uploaded
    if not captions:
        st.info("🚨 No captions loaded from backend.")
    else:
        st.info("👆 Upload an image to get caption recommendations")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit, Hugging Face Transformers, and CLIP")
