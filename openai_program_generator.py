from openai import OpenAI
import os
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import pdb
import re
import pprint
import time
import random
import string

# Replace 'YOUR_API_KEY' with your actual OpenAI API key
api_key=''
client = OpenAI(api_key=api_key)

delimiter = "-----------<this will be used to split on later>----------------"

system_prompt1 = '''
Take this example math problem below, and make a python program with a function called "def generate_math_problem()" that takes no arguments and that randomly generates a new version of that math word problem with two large changes to the math word problem.

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

input =
''' # Use your original system prompt text here

# Initialize the OpenAI client

# def test_if_compiles(full_code, n_examples=25):
#     examples = []
#     try:
#         exec(full_code, globals())
#         for i in range(n_examples):
#             example = generate_math_problem()
#             examples.append(example)
#         return examples, True, ""
#     except SyntaxError as e:
#         print(f"Syntax error in the code: {e}")
#         print(f"Error occurred on line {e.lineno}")
#         lines = full_code.split('\n')
#         if 0 <= e.lineno - 1 < len(lines):
#             print(f"Problematic line: {lines[e.lineno - 1]}")
#         return examples, False, str(e)
#     except Exception as e:
#         print(f"An error occurred while executing the code: {e}")
#         return examples, False, str(e)


# def test_if_compiles(full_code, n_examples=25):
#     examples = []
#     try:
#         exec(full_code, globals())
#         with ThreadPoolExecutor(max_workers=1) as executor:
#             for i in range(n_examples):
#                 future = executor.submit(generate_math_problem)
#                 try:
#                     example = future.result(timeout=3)  # Timeout after 3 seconds
#                     examples.append(example)
#                 except TimeoutError:
#                     print("Timeout: generate_math_problem() took longer than 3 seconds.")
#                     return examples, False, "TimeoutError: generate_math_problem() took too long."
#         return examples, True, ""
#     except SyntaxError as e:
#         print(f"Syntax error in the code: {e}")
#         print(f"Error occurred on line {e.lineno}")
#         lines = full_code.split('\n')
#         if 0 <= e.lineno - 1 < len(lines):
#             print(f"Problematic line: {lines[e.lineno - 1]}")
#         return examples, False, str(e)
#     except Exception as e:
#         print(f"An error occurred while executing the code: {e}")
#         return examples, False, str(e)
from multiprocessing import Process, Queue
import time

def run_code_and_generate_problems(full_code, n_examples, queue):
    try:
        # Execute the provided code
        exec(full_code, globals())
        # Generate examples using the dynamically defined function
        examples = []
        for _ in range(n_examples):
            example = generate_math_problem()  # This should now work because it's in the same process
            examples.append(example)
        
        queue.put((examples, True, ""))  # Return successful results

    except SyntaxError as e:
        error_msg = f"Syntax error in the code: {e}\nError occurred on line {e.lineno}"
        lines = full_code.split('\n')
        if 0 <= e.lineno - 1 < len(lines):
            error_msg += f"\nProblematic line: {lines[e.lineno - 1]}"
        queue.put(([], False, error_msg))

    except Exception as e:
        queue.put(([], False, f"An error occurred while executing the code: {str(e)}"))

def test_if_compiles(full_code, n_examples=25):
    queue = Queue()
    process = Process(target=run_code_and_generate_problems, args=(full_code, n_examples, queue))
    process.start()
    process.join(timeout=3)  # Timeout after 3 seconds

    if process.is_alive():
        print("Timeout: Code execution took longer than 3 seconds.")
        process.terminate()  # Forcefully kill the process
        process.join()
        return [], False, "TimeoutError: Code execution took too long."

    if not queue.empty():
        return queue.get()
    else:
        return [], False, "Unknown error: The process did not return any output."



# def get_openai_response(prompt, system_prompt):
#     chat_completion = client.chat.completions.create(
#         messages=[
#             {
#                 "role": "user",
#                 "content": system_prompt + prompt
#             }
#         ],
#         model="gpt-4",
#     )

#     message = chat_completion.choices[0].message.content
#     print(message)
#     return prompt, message.strip()
def get_openai_response(prompt, system_prompt, max_retries=10, wait_between_retries=20):
    attempts = 0
    while attempts < max_retries:
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": system_prompt + prompt
                    }
                ],
                model="gpt-4",
            )
        
            message = chat_completion.choices[0].message.content
            print(message)
            return prompt, message.strip()
        except openai.OpenAIError as e:
            print(f"An error occurred: {e}")
            attempts += 1
            if attempts < max_retries:
                print(f"Retrying in {wait_between_retries} seconds...")
                time.sleep(wait_between_retries)
            else:
                print("Max retries reached. Unable to fetch response.")
                raise
                
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise

def validate_sample(sample):
    system_prompt = "You are a math problem validator. Given a question and an answer, respond with '<code>True</code>' if the answer correctly solves the question, and otherwise respond with '<code>False</code> because <insert reason>' where you provide the reason it is wrong. Do not be too strict. If the error is just a small rounding error or if the answer results in values being negative or a fraction that in reality can not be negative or a fraction (like a stick can not be negative or a fraction of a stick), that is ok, just respond with '<code>True</code>'. "

    prompt = f"Question: {sample['question']}\nAnswer: {sample['answer']}\nIs this answer correct?"

    _, response = get_openai_response(prompt, system_prompt)
    if '<code>' in response:
        valid = parse_code_blocks_from_response(response)
    else:
        valid = response
    
    return valid.lower() == 'true', response

def parse_code_blocks_from_response(codee):
    pattern = re.compile(r'<code>(.*?)</code>', re.DOTALL)
    matches = pattern.findall(codee)
    if len(matches)!=1:
        print(codee)
        raise Exception(f"matches not equal to 1, OpenAI not following code block pattern correctly: {matches}")

    program_text = matches[0].strip()
    full_code = program_text.strip()
    return full_code


def try_to_fix_a_few_times(original_prompt, error_prompt,full_code,system_prompt1):
    attempts = 0
    retry_attempts = 5
    while attempts <= retry_attempts:
        attempts += 1
        _, fixed_response = get_openai_response(error_prompt, system_prompt1)
        
        fixed_response_ = fixed_response

        if "<code>" not in fixed_response_:
            result = False
            error = "There are no <code> </code> blocks in your answer, please try again."
        else:
            full_code = parse_code_blocks_from_response(fixed_response_)
            samples, result, error = test_if_compiles(full_code)
        if result:
            print("Fixed code ran successfully. Validating sample...")
            sample = random.choice(samples)
            if type(sample)=="NoneType":
                valid = False
                full_reply = "The function does not return anything per the example program. Please ensure you return {'question': question, 'answer': answer}"
            else:
                valid, full_reply = validate_sample(sample)
            if valid:
                print("Sample validated successfully. Writing to file.") 
                with open(os.path.expanduser("./output.txt"), 'a') as f:
                    f.write(f"\n{full_code}\n\n {delimiter}\n")
                return True
            else:
                print("Sample validation failed. Trying again... full_reply:" + full_reply)
                print("Fixed sample validation failed. Attempts:" + str(attempts))
                error_prompt=f"The program you created does not produce correct answers, please fix it. Here is an example of it producing bad samples: {sample}, is wrong and here is why: {full_reply}. Please fix the code and provide the corrected version:\n\n{full_code} for this word problem {original_prompt}"
        else:
            print("Fixed code still fails. Attempts:" + str(attempts))
            error_prompt = f"The following code produced an error: {error}\nPlease fix the code and provide the corrected version:\n\n{full_code} for this word problem {original_prompt}"
            print(error)

    print("tried 3 times.. giving up.. on: "+ original_prompt)
    print("x"*100)
    return False
    
def main():
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    # dataset = dataset.shuffle()
    

    n_workers = 5 # how many do do at one time
    user_prompts = [str(i) for i in dataset]
    starting_idx = 205+230+1485+2650+1800
    user_prompts = user_prompts[starting_idx:]
    num_batches =  len(user_prompts) // n_workers
    fail_counter = 0
    success_counter = 0
    print("creating tasks..")
    for j in range(num_batches+1):
        start_idx = j * n_workers
        end_idx = min(start_idx + n_workers, len(user_prompts))
        user_prompts_subset = user_prompts[start_idx:end_idx]
        print("-"*50)
        print(f"starting_idx:{starting_idx} start_idx:{start_idx} end_idx:{end_idx}")
        
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(get_openai_response, prompt, system_prompt1) for prompt in user_prompts_subset]
            responses = []
            
            try:
                for future in as_completed(futures, timeout=300):  # Timeout after 300 seconds (5 minutes)
                    prompt, response = future.result()
                    responses.append((prompt, response))
            except TimeoutError:
                print("Timeout: Not all tasks completed within 5 minutes.")  
                print(f"_starting_idx:{starting_idx} start_idx:{start_idx} end_idx:{end_idx}")
                raise
        for i, response in enumerate(responses, 1):
          try:
            print('-'*50)
            print(i)
    
            print('testing:')
            prompt = response[0]
            print('prompt: ',prompt)
            codee = response[1]
            success = False
            if "<code>" not in codee:
                result = False
                error_prompt = f"There are no <code> </code> blocks in your answer, please try again. Here is your code {codee} for word problem {prompt}"
                success = try_to_fix_a_few_times(prompt, error_prompt, codee, system_prompt1)
                if success:
                   success_counter+=1 
                else:
                    fail_counter+=1
                    
                continue
            else:
                full_code = parse_code_blocks_from_response(codee)
                print("full_code:", full_code)
                samples, result, error = test_if_compiles(full_code)
                
            
            if result:
                print("successfully ran! doing further testing on samples:")
                sample = random.choice(samples)
                print("sample:",sample)
                if type(sample)=="NoneType":
                    valid = False
                    full_reply = "The function does not return anything per the example program. Please ensure you return {'question': question, 'answer': answer}"
                else:
                    valid, full_reply = validate_sample(sample)
                if valid:
                    print("Sample validated successfully. Writing program to file.")
                    print("o"*50)
                    success=True
                    with open(os.path.expanduser("./output.txt"), "a") as f:
                        f.write(f"\n{full_code}\n\n {delimiter}\n")
                else:
                    print("Sample validation failed. Trying to fix... full_reply:" + full_reply)
                    error_prompt=f"The program you created does not produce correct answers, please fix it. Here is an example of it producing bad samples: {sample}, is wrong and here is why: {full_reply}. Please fix the code and provide the corrected version:\n\n{full_code} for this word problem {prompt}"
                    
                    success = try_to_fix_a_few_times(prompt, error_prompt, full_code, system_prompt1)
                    
            else:
                print("!!!!failed:")
                print(error)
                print("Attempting to fix the error...")
                error_prompt = f"The following code produced an error: {error}\nPlease fix the code and provide the corrected version:\n\n{full_code} for this word problem {prompt}"
                
                success = try_to_fix_a_few_times(prompt, error_prompt,full_code,system_prompt1)
            if success:
               success_counter+=1 
            else:
                fail_counter+=1
            print("xx"*50)
            print("fail_counter:"+str(fail_counter))
            print("success_counter:"+str(success_counter))
            print("xx"*50)
          except Exception as e:
            print("BIG FAIL:" + str(e) +" response:"+ str(response))
            fail_counter+=1
            continue


if __name__ == "__main__":
    print("starting..")
    main()
