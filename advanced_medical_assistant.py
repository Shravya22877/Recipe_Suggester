# interactive_medical_assistant.py

from langgraph.graph import StateGraph, START, END
from typing import TypedDict
import google.generativeai as ai

# --- Configure API key ---
ai.configure(api_key="YOUR_API_KEY")  # Replace with your own key

model = ai.GenerativeModel('gemini-2.0-flash')

# --- Define state ---
class State(TypedDict):
    symptoms: str
    medical_info: str
    recommendations: str
    home_care_tips: str
    specialist: str

# --- Step 1: Analyze Symptoms ---
def analyze_symptoms(state: State) -> dict:
    prompt = f"""
I have the following symptoms: {state['symptoms']}.
Provide a list of possible medical conditions or relevant information.
"""
    response = model.generate_content(prompt)
    state['medical_info'] = response.text
    return state

# --- Step 2: Next Steps ---
def suggest_next_steps(state: State) -> dict:
    prompt = f"""
Based on these symptoms: {state['symptoms']} and possible conditions:
{state['medical_info']}
Provide general recommendations and precautions.
(Informational only, not medical advice)
"""
    response = model.generate_content(prompt)
    state['recommendations'] = response.text
    return state

# --- Step 3: Home Care ---
def provide_home_care(state: State) -> dict:
    prompt = f"""
Given symptoms: {state['symptoms']} and possible conditions: {state['medical_info']},
suggest simple home care or self-help tips.
"""
    response = model.generate_content(prompt)
    state['home_care_tips'] = response.text
    return state

# --- Step 4: Specialist ---
def suggest_specialist(state: State) -> dict:
    prompt = f"""
Based on symptoms: {state['symptoms']} and possible conditions: {state['medical_info']},
suggest the type of medical specialist to consult.
"""
    response = model.generate_content(prompt)
    state['specialist'] = response.text
    return state

# --- Build agent graph ---
builder = StateGraph(State)
builder.add_node("analyze_symptoms", analyze_symptoms)
builder.add_node("suggest_next_steps", suggest_next_steps)
builder.add_node("provide_home_care", provide_home_care)
builder.add_node("suggest_specialist", suggest_specialist)

builder.add_edge(START, "analyze_symptoms")
builder.add_edge("analyze_symptoms", "suggest_next_steps")
builder.add_edge("suggest_next_steps", "provide_home_care")
builder.add_edge("provide_home_care", "suggest_specialist")
builder.add_edge("suggest_specialist", END)

graph = builder.compile()

# --- Interactive Menu ---
state = {
    "symptoms": "",
    "medical_info": "",
    "recommendations": "",
    "home_care_tips": "",
    "specialist": ""
}

print("Welcome to the Interactive Agentic Medical Assistant!")

while True:
    print("\nMenu:")
    print("1. Enter symptoms / query")
    print("2. Get medical information / possible conditions")
    print("3. Get general recommendations / next steps")
    print("4. Get home care tips")
    print("5. Get suggested specialist")
    print("6. Exit")

    choice = input("Choose an option (1-6): ")

    if choice == "1":
        state['symptoms'] = input("Enter your symptoms: ")
        # Update medical info immediately
        state = graph.invoke(state)
        print("Symptoms recorded and analyzed.")
    elif choice == "2":
        if not state['medical_info']:
            print("Please enter symptoms first (Option 1).")
        else:
            print("\nMedical Information:\n", state['medical_info'])
    elif choice == "3":
        if not state['recommendations']:
            print("Please enter symptoms first (Option 1).")
        else:
            print("\nRecommendations / Next Steps:\n", state['recommendations'])
    elif choice == "4":
        if not state['home_care_tips']:
            print("Please enter symptoms first (Option 1).")
        else:
            print("\nHome Care Tips:\n", state['home_care_tips'])
    elif choice == "5":
        if not state['specialist']:
            print("Please enter symptoms first (Option 1).")
        else:
            print("\nSuggested Specialist:\n", state['specialist'])
    elif choice == "6":
        print("Exiting. Stay healthy!")
        break
    else:
        print("Invalid choice. Please select 1-6.")