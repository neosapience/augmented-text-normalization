from openai import OpenAI
import json
import re

def get_answer(query, system_prompt="", api_key=None):
    client = OpenAI(api_key=api_key)

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        # model = "gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
    )
    return completion.choices[0].message.content

def parse_script(text, lang="json"):
    pattern = f"```{lang}" + f"\n(.*?)\n```"
    code_blocks = re.findall(pattern, text, re.DOTALL)
    return "\n\n".join(code_blocks)

def json_correction(text, error, api_key):
    system_prompt = """You will be given a string-written JSON, but has typo in JSON syntax. You will be given what error is occurring. Correct the syntax so that it can be an input for ast.literal_eval. You output must be in JSON style, like:
```json
${corrected json}$
```"""
    response = get_answer(text + f"\nError: {error}", system_prompt=system_prompt, api_key=api_key)
    parsed = parse_script(response)
    return parsed