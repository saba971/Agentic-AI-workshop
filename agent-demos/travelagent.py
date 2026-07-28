import gradio as gr
from openai import OpenAI

# Connect to Ollama
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# Prompt template
def build_prompt(destination, days, budget, preferences):
    return f"""
You are a smart AI travel planner.

Create a detailed travel plan with:

Destination: {destination}
Days: {days}
Budget: {budget} INR
Preferences: {preferences}

Include:
- Day-wise itinerary
- Places to visit
- Food recommendations
- Budget breakdown
- Travel tips
"""

# Main function
def generate_plan(destination, days, budget, preferences):
    try:
        prompt = build_prompt(destination, days, budget, preferences)

        response = client.chat.completions.create(
            model="llama3.1:latest",  # you can change to mistral or llama3
            messages=[
                {"role": "system", "content": "You are an expert travel planner."},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"

# Gradio UI
with gr.Blocks() as app:
    gr.Markdown("## 🌍 AI Travel Planner (Ollama + Gradio)")

    with gr.Row():
        destination = gr.Textbox(label="Destination", placeholder="e.g. Goa")
        days = gr.Number(label="Days", value=3)
        budget = gr.Number(label="Budget (INR)", value=15000)

    preferences = gr.Textbox(
        label="Preferences",
        placeholder="e.g. beaches, nightlife, budget hotels"
    )

    btn = gr.Button("Generate Plan")

    output = gr.Textbox(label="Travel Plan", lines=20)

    btn.click(
        fn=generate_plan,
        inputs=[destination, days, budget, preferences],
        outputs=output
    )

app.launch()