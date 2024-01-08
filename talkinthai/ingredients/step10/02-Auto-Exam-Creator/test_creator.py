import os

from openai import OpenAI


client = OpenAI()
class TestGenerator:
    def __init__(self, topic, num_possible_answers, num_questions):
        self.topic = topic
        self.num_possible_answers = num_possible_answers
        self.num_questions = num_questions

        if self.num_questions > 6:
            raise ValueError("Attention! Generation of many questions might be expensive!")
        if self.num_possible_answers > 5:
            raise ValueError("More than 5 possible answers is not supported!")

    def run(self):
        prompt = self.create_prompt()
        system_prompt = self.create_system_prompt()
        if TestGenerator._verify_prompt(prompt):
            response = self.generate_quiz(system_prompt, prompt)
            return response.choices[0].message.content
        
        raise ValueError("Prompt not accepted.")


    def generate_quiz(self, system_promopt, prompt):
        response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": system_promopt},
                            {"role": "user", "content": prompt},
                        ]
        )

        return response
        

    @staticmethod
    def _verify_prompt(prompt):
        print(prompt)
        response = input("Are you happy with the prompt? (y/n)")

        if response.upper() == "Y":
            return True
        return False


    def create_prompt(self):
        prompt = f"Create a multiple choice quiz on the topic of {self.topic} consisting of {self.num_questions} questions. " \
                 + f"Each question should have {self.num_possible_answers} options. "\
                 + f"Also include the correct answer for each question using the starting string 'Correct Answer: '."
        return prompt
    
    def create_system_prompt(self):
        system_prompt = "You are a helpful assistant for test generation."
        return system_prompt
        

if __name__ == "__main__":
    gen = TestGenerator("Python", 4, 2)
    response = gen.run()
    print(response)
    