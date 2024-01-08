import os
from openai import OpenAI
client = OpenAI()

class RecipeGenerator:
    
    def __init__(self):
            
        
        self.list_of_ingredients = self.ask_for_ingredients()

    @staticmethod
    def ask_for_ingredients():
        list_of_ingredients = []
        while True:
            ingredient = input("Enter an ingredient (or type 'done' to finish): ")
            if ingredient.lower() == "done":
                break
            list_of_ingredients.append(ingredient)
        
        print(f"Your ingredients are: {', '.join(list_of_ingredients)}")
        
        return list_of_ingredients

    def generate_recipe(self):
        system_prompt = RecipeGenerator.create_system_prompt()
        prompt = RecipeGenerator.create_recipe_prompt(self.list_of_ingredients)
        if RecipeGenerator._verify_prompt(prompt):
            response = RecipeGenerator.generate(system_prompt, prompt)
            return response.choices[0].message.content
        raise ValueError("Prompt not accepted.")

    @staticmethod
    def create_system_prompt():
        return "You are a helpful cooking assistant."

    @staticmethod
    def create_recipe_prompt(list_of_ingredients):
        prompt = f"Create a detailed recipe based on only the following ingredients: {', '.join(list_of_ingredients)}.\n" \
                + f"Additionally, assign a title starting with 'Recipe Title: ' to this dish, which can be used to create a photorealistic image of it."
        return prompt

    @staticmethod
    def _verify_prompt(prompt):
        print(prompt)
        response = input("Are you happy with the prompt? (y/n)")

        if response.upper() == "Y":
            return True
        return False

    @staticmethod
    def generate(system_prompt, prompt):
        response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                        ]
)
        return response

    def store_recipe(self, recipe, filename):
        with open(filename, "w") as f:
            f.write(recipe)


if __name__ == "__main__":
    """
    Test RecipeGenerator class without creating an image of the dish.
    """

    gen = RecipeGenerator()
    recipe = gen.generate_recipe()
    print(recipe)
