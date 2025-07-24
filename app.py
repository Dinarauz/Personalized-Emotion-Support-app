import pandas as pd
import gradio as gr
from sentence_transformers import SentenceTransformer, util
from datetime import datetime
import os

# 🔹 Load responses
df = pd.read_excel("responses.xlsx", engine="openpyxl")
df['keyword'] = df['keyword'].str.lower()

# 🔹 Load sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 🔹 Precompute keyword embeddings
keyword_embeddings = model.encode(df['keyword'].tolist(), convert_to_tensor=True)

# 🔹 Match + respond + log
def match_emotions(user_input, display_choice):
    input_embedding = model.encode(user_input, convert_to_tensor=True)
    similarity_scores = util.cos_sim(input_embedding, keyword_embeddings)[0]
    best_idx = similarity_scores.argmax().item()
    best_score = similarity_scores[best_idx].item()

    if best_score < 0.4:
        log_interaction(user_input, "No match", "-", "-", best_score)
        return "No strong match found. Try rephrasing."

    matched_row = df.iloc[best_idx]
    keyword = matched_row['keyword']
    response = matched_row['responses']
    action = matched_row['things_to_do']

    # Build output
    result = f"💡 Matched keyword: **{keyword}**\n\n"
    if display_choice == "Response":
        result += f"🗣️ Response: {response}"
    elif display_choice == "Things to do":
        result += f"🧭 Things to do: {action}"
    else:
        result += f"🗣️ Response: {response}\n\n🧭 Things to do: {action}"

    # Log interaction (in memory only if hosted)
    log_interaction(user_input, keyword, response, action, best_score)

    return result

# 🔹 Log function (optional: skip saving if on Hugging Face)
def log_interaction(user_input, keyword, response, action, score):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_input": user_input,
        "matched_keyword": keyword,
        "response": response,
        "things_to_do": action,
        "similarity_score": score
    }

    log_df = pd.DataFrame([log_entry])
    try:
        existing = pd.read_csv("interaction_log.csv")
        log_df = pd.concat([existing, log_df], ignore_index=True)
    except FileNotFoundError:
        pass

    # Only save if write permissions available
    try:
        log_df.to_csv("interaction_log.csv", index=False)
    except Exception:
        pass

# 🔹 Gradio UI
gr.Interface(
    fn=match_emotions,
    inputs=[
        gr.Textbox(label="How is she feeling?"),
        gr.Radio(["Response", "Things to do", "Both"], label="What would you like to see?", value="Both")
    ],
    outputs="text",
    title="Personality Assistant (Manual)",
    description="Type how she's feeling. The assistant will understand and support you based on emotion."
).launch()