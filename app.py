import streamlit as st
import openai
import numpy as np
import faiss
from dotenv import load_dotenv
import os

# Load OpenAI API key from .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Ensure the API key is set
if not api_key:
    raise ValueError("🚨 OPENAI_API_KEY is missing. Check your .env file or set it manually.")

# Initialize OpenAI client
client = openai.OpenAI(api_key=api_key)

# Sample holistic health-related documents
documents = [
    'Drinking warm lemon water in the morning helps detoxify your liver and improves digestion.',
    'Grounding, or walking barefoot on natural surfaces, can help reduce inflammation and improve sleep quality.',
    'Intermittent fasting can boost mitochondrial function and support cellular repair.',
    'Chamomile tea is known for its calming effects and can help with sleep and anxiety reduction.',
    'Exposing yourself to morning sunlight helps regulate your circadian rhythm and improves mood.',
    'Cold showers can boost circulation, improve immune function, and increase mental alertness.',
    'Eating fermented foods like kimchi and yogurt supports gut microbiome health.',
    'Practicing gratitude daily can reduce stress and improve overall mental health.',
    'Turmeric has powerful anti-inflammatory properties and supports joint health.',
    'Deep breathing exercises help lower cortisol levels and promote relaxation.',
    'Drinking green tea can enhance brain function and metabolic rate.',
    'A diet rich in omega-3 fatty acids supports heart and brain health.',
    'Magnesium deficiency can lead to muscle cramps and sleep disturbances.',
    'Spending time in nature can reduce anxiety and improve overall well-being.',
    'Getting enough sleep is crucial for hormone regulation and cognitive function.',
    'Probiotics support gut health by balancing beneficial bacteria.',
    'Blue light exposure at night disrupts melatonin production and sleep quality.',
    'Seed cycling can help regulate estrogen and progesterone balance in women.',
    'Sweating through exercise or sauna supports detoxification by eliminating heavy metals.',
    'Cold exposure can stimulate mitochondrial biogenesis, improving energy production.',
    'Journaling can improve mental clarity and emotional well-being.',
    'Avocados, MCT oil, and omega-3 and omega-9 fats promote satiety and help reduce belly fat.',
    'Berries are rich in antioxidants and fiber, helping to reduce oxidative stress and support gut health.',
    'Chia seeds expand in the stomach, promoting fullness due to their fiber and healthy fat content.',
    'Regular exercise reduces the risk of depression and anxiety while enhancing cognitive function.',
    'Quality sleep improves insulin sensitivity and regulates appetite-controlling hormones.',
    'Nuts and seeds are packed with protein, fiber, and essential vitamins, reducing heart disease risk.',
    'Exposure to natural light regulates circadian rhythm and enhances mood stability.',
    'Coffee is rich in antioxidants and may lower the risk of Parkinson’s and Alzheimer’s diseases.',
    'Fatty fish provide omega-3 fatty acids that support heart and brain health and reduce dementia risk.',
    'Avoiding ultra-processed foods lowers the risk of obesity and type 2 diabetes.',
    'Consuming probiotic-rich yogurt helps balance gut microbiota and improve digestion.',
    'Green tea improves cognitive function and boosts metabolism naturally.',
    'Daily meditation helps manage stress and improve concentration levels.',
    'Spending time in nature reduces anxiety and promotes overall well-being.',
    'Staying hydrated is essential for optimal body functions and energy levels.',
    'Holistic health approaches emphasize the integration of mind, body, and spirit to promote overall well-being.',
    'Incorporating spiritual care into nursing practices can enhance patient outcomes and satisfaction.',
    'Traditional healing practices often involve a holistic approach, addressing physical, emotional, and spiritual aspects of health.',
    "Holistic nursing care involves considering patients' psychological and social needs alongside their physical health.",
    'Complementary therapies, such as meditation and acupuncture, are integral components of holistic health practices.',
    'Intermittent fasting enhances autophagy, a process that clears damaged cells and improves mitochondrial efficiency.',
    'Cold exposure stimulates mitochondrial biogenesis, enhancing energy production and metabolic efficiency.',
    'Chronic inflammation is linked to mitochondrial dysfunction, leading to oxidative stress and cellular aging.',
    'Circadian rhythm misalignment can increase the risk of metabolic disorders, including insulin resistance and obesity.',
    'A balanced gut microbiome is crucial for neurotransmitter production, impacting mood and cognitive function.',
    'Omega-3 fatty acids play a vital role in reducing neuroinflammation and supporting brain plasticity.',
    'Vitamin D is essential for immune regulation and has been linked to improved mitochondrial function.',
    'Exposure to early morning sunlight helps synchronize the circadian clock, improving sleep quality and hormonal balance.',
    'Ketogenic diets have been shown to enhance mitochondrial function by increasing beta-oxidation of fatty acids.',
    'Polyphenols found in berries and green tea have antioxidant properties that protect mitochondria from oxidative damage.',
    'Infrared sauna therapy can promote detoxification and improve mitochondrial energy production.',
    'Regular exercise increases mitochondrial density, improving endurance and overall metabolic health.',
    'Melatonin is not only a sleep regulator but also a potent antioxidant that protects mitochondrial DNA from oxidative stress.',
    'Excessive blue light exposure at night suppresses melatonin production, disrupting sleep and circadian regulation.',
    'Probiotic-rich foods like kefir and sauerkraut help maintain gut microbiome diversity, supporting digestion and immune function.'
]

# Function to generate embeddings using OpenAI
def get_embeddings(texts):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=texts
    )
    return [embedding.embedding for embedding in response.data]

# Convert documents into embeddings and create FAISS index (only runs once)
@st.cache_resource
def build_faiss_index():
    document_embeddings = get_embeddings(documents)
    embedding_dim = len(document_embeddings[0])  # OpenAI's model produces 1536-dimension vectors

    # Initialize FAISS index and add embeddings
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(np.array(document_embeddings).astype('float32'))
    
    return index, document_embeddings

index, document_embeddings = build_faiss_index()

# Function to process user queries and return relevant AI-generated responses
def generate_optimized_response(query, top_k=5):
    query_embedding = get_embeddings([query])

    # Retrieve the closest documents using FAISS
    distances, indices = index.search(np.array(query_embedding).astype('float32'), top_k)

    # Sort and return only the most relevant documents
    sorted_docs = [documents[i] for i in indices[0]]

    prompt = (
        "Use the following holistic health insights to answer the user's question. "
        "If no relevant information is available, rely on your own expertise:\n\n"
        f"{' '.join(sorted_docs)}\n\n"
        f"User query: {query}\n\n"
        "Provide a structured and scientifically accurate response."
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a holistic health expert. Provide responses in a scientific, well-structured manner."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200
    )

    return response.choices[0].message.content

# Streamlit Web Interface
st.title("Holistic Health AI Assistant")
st.write("Ask a question about holistic health, functional medicine, or biohacking!")

query = st.text_input("Enter your question:")
if st.button("Get Answer"):
    response = generate_optimized_response(query)
    st.write("### AI Response:")
    st.success(response)
