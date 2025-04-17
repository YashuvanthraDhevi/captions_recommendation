# captions_recommendation
AI Image Caption Recommendation System

Applications like Instagram still lack a caption-generation tool because most LLMs don’t write captions; they describe the image instead. This is where a caption recommendation system can help, using a library of captions to find the caption that goes well with the image. 

An AI Image Caption Recommendation System using a retrieval-based approach, leveraging CLIP (Contrastive Language–Image Pre-training) for both image and text understanding is built.

First, the input image will be preprocessed and fed into the CLIP image encoder to generate a high-dimensional feature vector representing the image’s visual content. A similar process will be applied to a set of candidate captions, using the CLIP text encoder to generate text embeddings for each caption. Next, the cosine similarity between the image embedding and each caption embedding will be calculated. This similarity score will quantify how well each caption aligns with the visual content of the image.

Finally, the system ranks the candidate captions based on their similarity scores and presents the top-ranked captions as recommendations.

Find out best captions for your image using this app.

App Link: https://captionsrecommendation-yashu.streamlit.app/
