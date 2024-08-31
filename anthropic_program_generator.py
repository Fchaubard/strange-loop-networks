import anthropic
import os
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
import pdb
import re
import pprint

# Replace 'YOUR_API_KEY' with your actual Anthropic API key
api_key=''
delimiter = "-----------<this will be used to split on later>----------------"

system_prompt1 = '''Take this example math problem below, and make a python program with a function called "def generate_math_problem()" that takes no arguments and that randomly generates a new version of that math word problem with two large changes to the math word problem.

First, the numbers in the math word problem should be randomly generated, but still gives the right answers to the original problem. You should replace all the "input numbers" into the problem with random positive numbers that are in the range plus or minus 150% of the original value in the problem. But you must also ensure the random numbers have the same significant figures and precision as the original values (should be an integer if the word problem uses integers, and 2 digit floats if the word problem uses 2 digit floats). You should never have the program output more than 2 decimal points for any number the program outputs (so you should never output something like this: 3.8500000000000005). If the numbers are text based like "three" or "fourteen" you should replace those as well with a random int if it makes sense to do so. 

Second, you should also replace the names of all the people in the problem, and it should still make sense. You should replace the names with a random sequence of chars in the alphabet. You should never replace anything but the names of people, so you should never replace words like "the" or "is" or "car".

So for example, if the example math word problem says "question: Tobias is buying a new pair of shoes that costs $95. Tobias has $32 left after the purchase. How much did Tobias start with? answer: Since Tobias had $32 left after purchasing the $95 shoes, he had <$95+$32=$127> $127 to start. #### 127" you should generate a program that can output something like this: "question: asfdlkjaBBasdfD723 is buying a new pair of shoes that costs $87. asfdlkjaBBasdfD723 has $23 left after the purchase. How much did asfdlkjaBBasdfD723 start with? answer: Since asfdlkjaBBasdfD723 had $23 left after purchasing the $87 shoes, he had <$23+$87=$100> $100 to start. #### 100" It would be incorrect to output "asfdlkjaBBasdfD723 is buying a new pair of shoes that costs $32.322" because first, the replacement name must only include chars in the alphabet but also the number of significant digits has increased. You should wrap your answer numbers in a round() function to prevent such issues like irrational numbers and repeating decimal places like this: 59/72=0.8194444444444444. And instead should output 59/72=0.8194. 

The program should take in as input this line of text below (input) and print the output to stdout the new math problem, which will be a new augmented question + answer. Ensure you follow THE EXACT conventions of the example math word problem. DO NOT USE ANY LINE CONTINUATION CHARACTERS NO MATTER HOW LONG THE LINE IS IN YOUR PROGRAM!  YOU MUST WRAP YOUR OUTPUT CODE IN YOUR RESPONSE IN <code> </code> BLOCKS LIKE SO: <code>def f(): return 7</code> and we will extract all plain-text in between the code blocks and compile it directly. Please match the output format which should be a dict with 2 keys: 'question' and 'answer'. Do NOT put more than one <code> </code> block in your response as we expect only one. Here is an example math word problem and program for you to learn the task. 

Example math word problem: 

input = {'question': 'Chrystal’s vehicle speed is 30 miles per hour. Ascending the mountain decreases its speed by fifty percent, and descending the mountain increases its speed by twenty percent. If the distance going to the top of the mountain is 60 miles and the distance going down to the foot of the mountain is 72 miles, how many hours will Crystal have to pass the whole mountain?', 'answer': "The vehicle's speed decreases to 30 x 0.50 = <<30*0.50=15>>15 miles per hour when ascending to the top of the mountain.\ nSo, the total time Crystal will have to spend going to the top of the mountain is 60 / 15 = <<60/15=4>>4 hours.\nAnd the speed of the vehicle increases by 30 x 0.20 = <<30*0 .20=6>>6 miles per hour when going down to the foot of the mountain.\nSo, the total speed of the vehicle when going down is 30 + 6 = <<30+6=36>>36 miles per hour.\nThus, Chry stal will have to spend 72 / 36 = <<72/36=2>>2 hours to descend from the mountain.\nTherefore, the total hours she will spend to pass the whole mountain is 4 + 2 = <<4+2=6>>6 hours.\n#### 6"}

Example python program that you should create from the above math word problem:
<code>
import random  
import string             
                          
def generate_math_problem():                                                                        
    # Generate random numbers for speed, ascending speed change, descending speed change, ascending distance and descending distance                                
    speed = random.randint(15,45)  # The current speed at 30 mph is replaced by a random integer between +/- 50% of the speed                                                 
    asc_percent = round(random.random()*.5*1.5) ,2)              # It's a percentage and plus or minus 50%, and we do not want to go negative
    desc_percent = round(random.random()*.3) ,2)             # Likewise it's a percentage 
    asc_dist = random.randint(30,90) # The ascending distance of 60 miles is replaced by a random int between +/- 50%                                                         
    desc_dist = random.randint(36,108) # The descending distance of 72 miles is replaced by a random int between +/- 50%                                                      

    # Generate snake_case random string of length 7 for name                                        
    name = ''.join(random.choice(string.ascii_lowercase) for _ in range(7))                         
                          
    question = f"{name.capitalize()}'s vehicle speed is {speed} miles per hour. Ascending the mountain decreases its speed by {asc_percent} percent, and descending the mountain increases its speed by {desc_percent} percent. If the distance going to the top of the mountain is {asc_dist} miles and the distance going down to the foot of the mountain is {desc_dist} miles, how many hours will {name.capitalize()} have to pass the whole mountain?"                    
                          
    answer = f"The vehicle's speed decreases to {speed} x {asc_percent} = {speed * asc_percent} miles per hour when ascending to the top of the mountain.\nSo, the total time {name.capitalize()} will have to spend going to the top of the mountain is {asc_dist} / {speed * asc_percent} = { round(asc_dist / (speed * asc_percent), 2) } hours.\nAnd the speed of the vehicle increases by {speed} x {desc_percent} = { round(speed * desc_percent,2) } miles per hour when going down to the foot of the mountain.\nSo, the total speed of the vehicle when going down is {speed} + {speed * desc_percent} = {speed + speed * desc_percent} miles per hour.\nThus, {name.capitalize()} will have to spend {desc_dist} / {speed + speed * desc_percent} = { round(desc_dist / (speed + speed * desc_percent),2)} hours to descend from the mountain.\nTherefore, the total hours she will spend to pass the whole mountain is { round(asc_dist / (speed * asc_percent),2) } + { round(desc_dist / (speed + speed * desc_percent),2)} = {round(asc_dist / (speed * asc_percent),2) + round(desc_dist / (speed + speed * desc_percent),2)} hours.\n#### {round(asc_dist / (speed * asc_percent),2) + round(desc_dist / (speed + speed * desc_percent),2)}"
    return {'question': question, 'answer': answer}
generate_math_problem()           
</code> 


Now for your math word problem. Please follow a similar format as per the example program above in your program:

input = '''

# Initialize the Anthropic client
client = anthropic.Anthropic(api_key=api_key)

def test_if_compiles(full_code, n_examples=25):
    examples = []
    try:
        compile(full_code, '<string>', 'exec')
        exec(full_code, globals())
        for i in range(n_examples):
            example = generate_math_problem()
            examples.append(example)
        return examples, True, ""
    except SyntaxError as e:
        print(f"Syntax error in the code: {e}")
        print(f"Error occurred on line {e.lineno}")
        lines = full_code.split('\n')
        if 0 <= e.lineno - 1 < len(lines):
            print(f"Problematic line: {lines[e.lineno - 1]}")
        return examples, False, str(e)
    except Exception as e:
        print(f"An error occurred while executing the code: {e}")
        return examples, False, str(e)

def get_claude_response(prompt, system_prompt):
    message = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=2000,
        system=system_prompt,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return prompt, message.content

def validate_sample(sample):
    system_prompt = "You are a math problem validator. Given a question and an answer, respond with 'True' if the answer correctly solves the question, and 'False' otherwise. Respond with only 'True' or 'False'. If you are unsure, or if the question or answer do not make sense, just respond with 'False'."
    
    prompt = f"Question: {sample['question']}\nAnswer: {sample['answer']}\nIs this answer correct?"
    
    _, response = get_claude_response(prompt, system_prompt)
    
    return response[0].text.strip().lower() == 'true'

def main():
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    dataset = dataset.shuffle()
    dataset = dataset.select(range(3))
    user_prompts = [str(i) for i in dataset]
    print("creating tasks..")
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(get_claude_response, prompt, system_prompt1) for prompt in user_prompts]
        responses = []
        for future in as_completed(futures):
            prompt, response = future.result()
            responses.append((prompt, response))

    # Write responses to file
    with open(os.path.expanduser("./output.txt"), "w") as f:
        for i, response in enumerate(responses, 1):
            print('-'*50)
            print(i)
            
            print('testing:')
            prompt = response[0]
            print('prompt: ',prompt)
            codee = response[1][0].text
            
            pattern = re.compile(r'<code>(.*?)</code>', re.DOTALL)
            matches = pattern.findall(codee)
            if len(matches)!=1:
                print(codee)
                
                raise Exception(f"matches not equal to 1, claude not following code block pattern correctly: {matches}")
            
            program_text = matches[0].strip()
            full_code = program_text.strip()

            
            print("full_code:", full_code)
            samples, result, error = test_if_compiles(full_code)
            
            if result:
                print("successfully ran! doing further testing on samples:")
                print(samples[0])
                if validate_sample(samples[0]):
                    print("Sample validated successfully. Writing to file.")
                    
                    f.write(f"\n{full_code}\n\n {delimiter}\n")
                else:
                    print("Sample validation failed. Skipping this prompt.")
            else:
                print("!!!!failed:")
                print(error)
                print("Attempting to fix the error...")
                attempts = 0
                while attempts<=3:
                    attempts+=1
                    error_prompt = f"The following code produced an error: {error}\nPlease fix the code and provide the corrected version:\n\n{full_code}"
                    _, fixed_response = get_claude_response(error_prompt, system_prompt1)
                    fixed_response_ = fixed_response[0].text
                    fixed_matches = pattern.findall(fixed_response_)
                    if len(fixed_matches) != 1:
                        print("Failed to get a proper fixed response from Claude. Skipping this prompt.")
                        continue
                    
                    full_code = fixed_matches[0].strip()
                    samples, result, error = test_if_compiles(full_code)
                    
                    if result:
                        print("Fixed code ran successfully. Validating sample...")
                        if validate_sample(samples[0]):
                            print("Fixed sample validated successfully. Writing to file.")
                            f.write(f"\n{full_code}\n\n {delimiter}\n")
                            break
                        else:
                            print("Fixed sample validation failed. Skipping this sample. Attempts:" + str(attempts))
                    else:
                        print("Fixed code still fails. Attempts:" + str(attempts))
                        print(error)


                print("tried 3 times.. giving up.. on: "+ prompt)

if __name__ == "__main__":
    print("starting..")
    main()