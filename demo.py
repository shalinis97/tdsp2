import subprocess
import json

# Expected answers for validation
expected_answers = {
    "GA1 Q1": "Version: Code 1.97.2",
    "GA1 Q2": "",
    "GA1 Q3": "",
    "GA1 Q4": "",
    "GA1 Q5": "",
    "GA1 Q6": "",
    "GA1 Q7": "",
    "GA1 Q8": "",
    "GA1 Q9": "",
    "GA1 Q10": "",
    "GA1 Q11": "",
    "GA1 Q12": "",
    "GA1 Q13": "",
    "GA1 Q14": "",
    "GA1 Q15": "",
    "GA1 Q16": "",
    "GA1 Q17": "",
    "GA1 Q18": "",
    "GA2 Q5": "262036",
    "GA3 Q9": "Yes",
    "GA5 Q1": 0.2362,
    "GA5 Q2": 86,
    "GA5 Q3": 493,
    "GA5 Q4": 2762,
    "GA5 Q5": 13924,
    "GA5 Q6": 52584,
    "GA5 Q7": 26053,
    "GA5 Q8": "",
    "GA5 Q9": "",
    "GA5 Q10": ""
}

# List of curl commands to evaluate API responses
curl_commands = [
    # GA1 Q1 - Output of 'code -s' with variations
    ('GA1 Q1', '''curl -X POST "https://tdsp2.vercel.app/api/" \
    -H "Content-Type: multipart/form-data" \
    -F "question=What is the output of code -s?"'''),

    # GA2 Q5 - Number of pixels with lightness > 0.133 with variations
    ('GA2 Q5', '''curl -X POST "https://tdsp2.vercel.app/api/" \
    -H "Content-Type: multipart/form-data" \
    -F "question=What is the number of pixels with lightness > 0.133?" \
    -F "file=@lenna.webp"'''),

    # GA3 Q9 - LLM prompt to say Yes with variations
    ('GA3 Q9', '''curl -X POST "https://tdsp2.vercel.app/api/" \
    -H "Content-Type: multipart/form-data" \
    -F "question=Write a prompt that will get the LLM to say Yes."'''),
    ('GA3 Q9', '''curl -X POST "https://tdsp2.vercel.app/api/" \
    -H "Content-Type: multipart/form-data" \
    -F "question=Make the LLM respond with Yes."'''),   ##-----not working have to check

    # GA5 Q1 - Calculate total margin with variations
    ('GA5 Q1', '''curl -X POST "https://tdsp2.vercel.app/api/" \
    -H "Content-Type: multipart/form-data" \
    -F "question=What is the total margin for transactions before Fri Nov 25 2022 06:28:05 GMT+0530 (India Standard Time) for Theta sold in IN (which may be spelt in different ways)?" \
    -F "file=@q-clean-up-excel-sales-data.xlsx"'''),
    ('GA5 Q1', '''curl -X POST "https://tdsp2.vercel.app/api/" \
    -H "Content-Type: multipart/form-data" \
    -F "question=Calculate total margin for Theta in India before November 25, 2022." \
    -F "file=@q-clean-up-excel-sales-data.xlsx"'''),

    # GA5 Q6 - Calculate total sales from JSONL file with variations
    ('GA5 Q6', '''curl -X POST "https://tdsp2.vercel.app/api/" \
    -H "Content-Type: multipart/form-data" \
    -F "question=What is the total sales value?" \
    -F "file=@q-parse-partial-json.jsonl"'''),
    ('GA5 Q6', '''curl -X POST "https://tdsp2.vercel.app/api/" \
    -H "Content-Type: multipart/form-data" \
    -F "question=Calculate the total sales from the JSON file." \
    -F "file=@q-parse-partial-json.jsonl"''')
]

# Execute each curl command and print the output
for question, command in curl_commands:
    #print(f"Executing: {command}\n")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        print("Output:", result.stdout)
        #print("Error:", result.stderr)

        # Check if output is in the correct JSON format
        try:
            output_json = json.loads(result.stdout)
            if "answer" in output_json:
                print(f"✅ {question} - Output format is correct.")
                answer = output_json.get("answer", "")

                # Check if the answer is correct
                if answer.startswith(expected_answers[question]):
                    print(f"✅ {question} - Answer is correct.")
                else:
                    print(f"❌ {question} - Answer is wrong.")
            else:
                print(f"❌ {question} - Output format is wrong.")
        except json.JSONDecodeError:
            print(f"❌ {question} - Output is not valid JSON.")

    except Exception as e:
        print(f"Failed to execute command: {e}")
    print("\n" + "-" * 80 + "\n")
