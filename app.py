import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np

# ---------------------------------
# STREAMLIT UI
# ---------------------------------
st.set_page_config(page_title="Simple RAG Demo", layout="wide")
st.title("📘 AI Basics RAG (Fixed Content)")

# ---------------------------------
# DOCUMENT CONTENT (INBUILT)
# ---------------------------------
DOCUMENT = """
Q1. What is Artificial Intelligence?
Artificial Intelligence is a field of computer science focused on creating machines that can perform tasks requiring human intelligence.
These tasks include learning, reasoning, problem-solving, and decision-making.
AI systems aim to simulate intelligent behavior using algorithms and data.

Q2. What are the main types of Artificial Intelligence?
The main types of AI are Narrow AI, General AI, and Super AI.
Narrow AI is designed for specific tasks and is the only type currently in use.
General and Super AI are theoretical and do not yet exist in real-world applications.

Q3. What is Narrow AI?
Narrow AI is an AI system designed to perform a single specific task.
Examples include chatbots, recommendation systems, and facial recognition.
It operates within limited constraints and cannot perform beyond its defined purpose.

Q4. What is Machine Learning?
Machine Learning is a subset of AI that enables systems to learn from data.
Instead of being explicitly programmed, models improve through experience.
It is widely used in prediction, classification, and pattern recognition.

Q5. What is Deep Learning?
Deep Learning is a specialized branch of Machine Learning.
It uses neural networks with multiple hidden layers to process complex data.
Deep Learning is commonly applied in speech recognition and image analysis.

Q6. What is a Neural Network?
A neural network is a computational model inspired by the human brain.
It consists of interconnected nodes called neurons that process information.
Neural networks are the foundation of deep learning systems.

Q7. What is Natural Language Processing?
Natural Language Processing, or NLP, enables machines to understand human language.
It allows computers to analyze, interpret, and generate text or speech.
NLP is used in chatbots, translation tools, and voice assistants.

Q8. What is Computer Vision?
Computer Vision is a field of AI that enables machines to interpret visual data.
It allows systems to analyze images and videos to identify objects or patterns.
Applications include facial recognition, medical imaging, and self-driving cars.

Q9. What is Data in Artificial Intelligence?
Data is the core component used to train AI models.
It provides the examples from which systems learn patterns and relationships.
High-quality data is essential for building accurate and reliable AI systems.

Q10. What is Supervised Learning?
Supervised learning is a machine learning approach that uses labeled data.
The model learns by comparing its output with the correct answer.
It is commonly used for classification and regression tasks.

Q11. What is Unsupervised Learning?
Unsupervised learning works with unlabeled data.
The model identifies patterns or groupings without predefined outputs.
Clustering and association are common unsupervised learning techniques.

Q12. What is Reinforcement Learning?
Reinforcement learning involves an agent interacting with an environment.
The agent learns by receiving rewards or penalties based on actions.
This approach is widely used in robotics and game-playing AI.

Q13. What is an AI Model?
An AI model is a mathematical representation trained on data.
It learns patterns and relationships to make predictions or decisions.
Models are the core components of AI systems.

Q14. What is Training in AI?
Training is the process of feeding data into an AI model.
During training, the model adjusts its parameters to reduce errors.
Effective training improves the accuracy of predictions.

Q15. What is Overfitting?
Overfitting occurs when a model learns the training data too well.
It performs poorly on new or unseen data.
This reduces the model’s ability to generalize.

Q16. What is Underfitting?
Underfitting happens when a model is too simple.
It fails to capture important patterns in the data.
As a result, performance is poor on both training and test data.

Q17. What is an Algorithm?
An algorithm is a step-by-step procedure used to solve a problem.
In AI, algorithms guide how data is processed and learned.
Different algorithms serve different AI tasks.

Q18. What are Applications of Artificial Intelligence?
AI is used in healthcare, finance, education, and transportation.
It improves efficiency, accuracy, and decision-making.
Common examples include virtual assistants and recommendation systems.

Q19. What are Ethical Issues in AI?
Ethical issues in AI include bias, privacy concerns, and job displacement.
AI systems may reflect biases present in training data.
Responsible development is required to address these concerns.

Q20. What is the Future of Artificial Intelligence?
The future of AI includes smarter automation and improved human-AI collaboration.
AI is expected to enhance productivity across industries.
Ongoing research focuses on making AI more transparent and ethical.
"""


# ---------------------------------
# LOAD MODELS
# ---------------------------------
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    llm = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=200
    )
    return embedder, llm

embedder, llm = load_models()

# ---------------------------------
# BUILD VECTOR STORE
# ---------------------------------
chunks = DOCUMENT.split(".")
embeddings = embedder.encode(chunks)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# ---------------------------------
# QUESTION INPUT
# ---------------------------------
question = st.text_input("Ask a question")

if question:
    q_emb = embedder.encode([question])
    D, I = index.search(np.array(q_emb), k=2)

    context = ". ".join([chunks[i] for i in I[0]])

    prompt = f"""
Answer ONLY from the context below.
If the answer is not present, say:
"Content not available in the provided document."

Context:
{context}

Question:
{question}

Answer:
"""

    answer = llm(prompt)[0]["generated_text"]
    st.subheader("Answer")
    st.write(answer)
