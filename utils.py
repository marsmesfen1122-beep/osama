import qrcode
import speech_recognition as sr
import smtplib
from openai import OpenAI
import pandas as pd
from deep_translator import GoogleTranslator
from langdetect import detect

# Assume 'df', 'model' (for sklearn), 'embeddings' (for faiss) and 'index' (for faiss) are accessible if used directly
# or passed as arguments to functions.
# For this demonstration, we'll assume a global 'df' and 'model' (RandomForestRegressor).

# Placeholder for the sklearn model for predict_popularity
# In a real application, this model would be loaded or passed.
# model = RandomForestRegressor() # This would need to be trained
# For now, let's assume predict_popularity relies on a pre-trained `model` or takes feature values directly.

# Helper for `classify_difficulty` and `generate_tags`
def classify_difficulty(page_count, ratings_count):
    if page_count < 200 and ratings_count < 500:
        return "Beginner"
    elif page_count < 400:
        return "Intermediate"
    else:
        return "Advanced"

def generate_tags(row):
    tags = []

    if row["average_rating"] > 4:
        tags.append("Highly Rated")

    # Assuming 'categories' column exists
    if "python" in str(row["categories"]).lower():
        tags.append("Programming")

    if row["page_count"] > 400:
        tags.append("Long Read")

    return tags

# Assuming `search_books` from `engine.py` is imported or available
# from .engine import search_books # This would be the actual import

# To make this runnable without modifying current notebook state significantly,
# we'll assume a dummy search_books or pass a callable.
# In a proper module structure, search_books would be imported.

# Placeholder for a dummy search_books if engine is not fully set up in this context:
def dummy_search_books(query, top_k=5):
    # This is a dummy. In a real scenario, it would call the AIEngine search method.
    print(f"Dummy search for: {query}, top_k={top_k}")
    # You would need to make 'df' available or pass it to this function.
    # For now, let's return an empty DataFrame or a mock one.
    return pd.DataFrame([{'title': 'Mock Book 1', 'authors': 'Mock Author', 'categories': 'Mock Category'}])

# All utility functions:

def predict_popularity(rating, count, trained_model): # Pass the trained model
    return trained_model.predict([[rating, count]])[0]

def predict_late_return(days_borrowed):
    if days_borrowed > 14:
        return "HIGH RISK OF LATE RETURN"
    elif days_borrowed > 10:
        return "MEDIUM RISK"
    else:
        return "LOW RISK"

def generate_qr(book_id):
    img = qrcode.make(book_id)
    img.save(f"{book_id}.png")
    return f"{book_id}.png"

def voice_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
        return r.recognize_google(audio)

def send_email(to_email, message):
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login("your_email@gmail.com", "password")

    server.sendmail(
        "your_email@gmail.com",
        to_email,
        message
    )
    server.quit()

client = OpenAI(api_key="YOUR_API_KEY") # Needs an actual API key

def ask_ai(question):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":"You are a library assistant"},
            {"role":"user","content":question}
        ]
    )
    return response.choices[0].message.content

def smart_search(query, search_books_func=dummy_search_books):
    results = search_books_func(query, top_k=5)

    context = ""
    for _, row in results.iterrows():
        context += row['title'] + " " + str(row['description'])

    return context

def top_books(df_instance):
    import matplotlib.pyplot as plt # Move import inside function
    top = df_instance.sort_values('ratings_count', ascending=False).head(10)

    plt.plot(top['title'], top['ratings_count'])
    plt.xticks(rotation=90)
    plt.show()

def learning_path_generator(goal, top_k=10, search_books_func=dummy_search_books):
    results = search_books_func(goal, top_k)

    path = {
        "goal": goal,
        "roadmap": []
    }

    for i, (_, row) in enumerate(results.iterrows(), 1):
        path["roadmap"].append({
            "step": i,
            "book": row["title"],
            "author": row["authors"],
            "reason": row["categories"]
        })

    return path

def personalized_recommend(user_pref, df_instance):
    category = user_pref.get("category", "")
    min_rating = user_pref.get("min_rating", 3)

    results = df_instance[
        (df_instance["categories"].str.contains(category, na=False)) &
        (df_instance["average_rating"] >= min_rating)
    ]

    return results.sort_values("ratings_count", ascending=False).head(10)

def similar_books(book_title, embeddings, index, df_instance, top_k=5):
    idx = df_instance[df_instance["title"] == book_title].index[0]

    query_vector = embeddings[idx].reshape(1, -1)

    distances, ids = index.search(
        np.array(query_vector).astype("float32"),
        top_k
    )

    return df_instance.iloc[ids[0]][["title", "authors", "categories"]]

def rag_chatbot(question, search_books_func=dummy_search_books):
    results = search_books_func(question, top_k=5)

    context = ""
    for _, row in results.iterrows():
        context += f"""
Title: {row['title']}
Description: {row['description']}
Category: {row['categories']}
"""

    return {
        "question": question,
        "context_used": context,
        "answer": "Use this context in LLM (OpenAI/Gemini) for final response"
    }

def trending_books(df_instance):
    return df_instance.sort_values(
        ["ratings_count", "average_rating"],
        ascending=False
    ).head(10)[["title", "ratings_count", "average_rating"]]

# Multilingual features
SUPPORTED_LANGUAGES = {
    "en": "english",
    "am": "amharic",
    "fr": "french",
    "ar": "arabic"
}

def translate_text(text, source="auto", target="en"):
    if target == "en":
        return text

    return GoogleTranslator(
        source=source,
        target=target
    ).translate(text)


def multilingual_search(query, user_lang="en", top_k=5, search_books_func=dummy_search_books):
    # Step 1: Translate query → English (for FAISS)
    english_query = translate_text(query, target="en")

    # Step 2: Search normally
    results = search_books_func(english_query, top_k)

    # Step 3: Translate results back
    translated_results = []

    for _, row in results.iterrows():

        title = row["title"]
        category = row["categories"]

        if user_lang != "en":
            title = translate_text(title, target=user_lang)
            category = translate_text(str(category), target=user_lang)

        translated_results.append({
            "title": title,
            "authors": row["authors"],
            "category": category
        })

    return translated_results

def multilingual_chatbot(question, lang="en", search_books_func=dummy_search_books):
    # Translate question → English
    english_q = translate_text(question, target="en")

    books = search_books_func(english_q, top_k=5)

    response = "Recommended Books:\n\n"

    for _, row in books.iterrows():
        response += f"""
Title: {row['title']}
Author: {row['authors']}
Category: {row['categories']}
"""

    # Translate response back if needed
    if lang != "en":
        response = translate_text(response, target=lang)

    return response

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"
