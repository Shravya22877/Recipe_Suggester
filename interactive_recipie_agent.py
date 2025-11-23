# interactive_recipe_agent.py

from langgraph.graph import StateGraph, START, END
from typing import TypedDict
import google.generativeai as ai

# --- Configure your Google Gemini API key here ---
ai.configure(api_key="AIzaSyAARKSW0mEBorpnhAgAGee-3ESlcwmUmXg")  # Replace with your own key

# Initialize the model
model = ai.GenerativeModel('gemini-2.0-flash')

# --- Define the state structure ---
class State(TypedDict):
    ingredients: str
    preferences: str
    recipe: str
    shopping_list: str
    cooking_steps: str

# --- Step 1: Suggest recipes ---
def suggest_recipe(state: State) -> dict:
    prompt = f"""
I have the following ingredients: {state['ingredients']}.
My dietary preferences are: {state['preferences']}.
Suggest 3 recipes I can make with these ingredients.
"""
    response = model.generate_content(prompt)
    state['recipe'] = response.text
    return state

# --- Step 2: Generate shopping list ---
def generate_shopping_list(state: State) -> dict:
    prompt = f"""
Based on these ingredients: {state['ingredients']} and the chosen recipes:
{state['recipe']}
Create a shopping list for any missing ingredients.
"""
    response = model.generate_content(prompt)
    state['shopping_list'] = response.text
    return state

# --- Step 3: Generate cooking steps ---
def generate_cooking_steps(state: State) -> dict:
    prompt = f"""
I want to cook this recipe/these recipes: {state['recipe']}.
Using the ingredients I have: {state['ingredients']}, 
provide a detailed, step-by-step cooking guide for these recipes.
"""
    response = model.generate_content(prompt)
    state['cooking_steps'] = response.text
    return state

# --- Build the agent graph ---
builder = StateGraph(State)
builder.add_node("suggest_recipe", suggest_recipe)
builder.add_node("generate_shopping_list", generate_shopping_list)
builder.add_node("generate_cooking_steps", generate_cooking_steps)

builder.add_edge(START, "suggest_recipe")
builder.add_edge("suggest_recipe", "generate_shopping_list")
builder.add_edge("generate_shopping_list", "generate_cooking_steps")
builder.add_edge("generate_cooking_steps", END)

graph = builder.compile()

# --- Interactive Input ---
print("Welcome to the Agentic Recipe Assistant!")
ingredients = input("Enter the vegetables/ingredients you have (comma-separated): ")
preferences = input("Enter your dietary preferences (or leave blank): ")

# --- Invoke agent ---
result = graph.invoke({
    "ingredients": ingredients,
    "preferences": preferences,
    "recipe": "",
    "shopping_list": "",
    "cooking_steps": ""
})

# --- Print results ---
print("\nSuggested Recipes:\n", result['recipe'])
print("\nShopping List:\n", result['shopping_list'])
print("\nCooking Steps:\n", result['cooking_steps'])