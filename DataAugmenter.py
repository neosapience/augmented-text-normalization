from tqdm import tqdm
import json
import sys
import argparse
import time
import os
from queue import Queue
import threading
import re
from utils import get_answer, parse_script, json_correction

class DataAugmenter:
    def __init__(self, idx, args, text=None):
        self.idx = idx
        self.args = args
        self.text = text
        self.api_key = args.openai_api_key

    def augment_from_scratch(self):
        prompt = f"""Consider a situation that there is a text that includes a lot of words that cannot be pronounced directly, like abbreviations, acronyms, numbers, symbols, measures, units, phone number, decimals, dates, etc. Your job is to create a challenging set of sentences that include such confusing words. Create {self.args.sentence_per_generation} sentences. Output only the generated sentences, splitted by newline."""
        response = get_answer(prompt, api_key=self.api_key)
        sentences = [r.strip("\n ") for r in response.split("\n") if len(r.strip("\n ")) > 0]
        normalized_sentences = []
        if self.idx == 0:
            for sentence in tqdm(sentences):
                self.text = sentence
                normalized = self.augment()
                normalized_sentences.extend(normalized)
        else:
            for sentence in sentences:
                self.text = sentence
                normalized = self.augment()
                normalized_sentences.extend(normalized)
        return normalized_sentences

    def augment(self):
        system_prompt = """You will be given a raw sentence that may have randomly capitalized letters, and your job is to output a normalized version of each word, puctuations, numbers, symbols, etc. Following the rules below.
Rule 0: Make some chunks from the sentences based on the words' semiotic classes. For example, for a sentence "On 22 September 2015, I went home.", the expected chunks are "On" (PLAIN), "22 September 2015" (DATE), "," (PUNCT), "I" (PLAIN), "went" (PLAIN), "home" (PLAIN), "." (PUNCT).
Possible semiotic classes are:
- PUNCT: punctuation (like ".")
- DATE: year, month, day... (like "October")
- LETTERS: the words that must be read letter-by-letter (like "USA")
- CARDINAL: numeric values (like "750,000")
- MONEY: money (like "$150")
- DECIMAL: numbers with decimal points (like "1.7 million")
- MEASURE: a value with unit (like "60 km")
- TELEPHONE: phone number (like 83-7177-229-7)
Rule 1: The normalization must be based on natural verbalization of abbreviations, acronyms, numbers, symbols, units, etc. Remember that the verbalization must correspond to the pronunciation in real speech.
Rule 2: Any non-english numbers must be verbalized to english words, like "£21" -> "twenty one pounds", "Louis XVI" -> "louis the sixteenth", "World War II" -> "world war two".
Rule 3: If an abbreviation or an acronym is read letter-by-letter, you must convert it into splitted letters, like "USA" -> "u s a" "TVs" -> "t v's".
Rule 4: If an abbreviation or an acronym is pronounced as itself (not letter-by-letter), you must leave it unchanged, like "NATO" -> "nato", "BASIC" -> "basic". 
Rule 5: There may be randomly capitalized common words, so be cautious not to read them letter-by-letter, like "HELLO" -> "hello", "LOVE" -> "love"
Rule 6: A chunk that is silenced must be normalized as "sil". Remeber that most punctuations are silenced.
Rule 7: Even there is a typo in the sentence, do not fix it.
Carefully consider the real-world pronunciation of each chunk and check the rules accurately. Then output verbalization for each word in json format: ```json[\n    {\n        "original_chunk": <ORIGINAL CHUNK>\n        "normalized_chunk": <NORMALIZED CHUNK>\n        "semiotic_class": <SEMIOTIC_CLASS>\n    },\n    {\n        "original_chunk": <ORIGINAL CHUNK>\n        "normalized_chunk": <NORMALIZED CHUNK>\n        "semiotic_class": <SEMIOTIC_CLASS>\n    }\n...\n]\n```.
"""
        response = get_answer(self.text.strip(), system_prompt=system_prompt, api_key=self.api_key)
        if not response:
            return None
        response = parse_script(response).strip("\n")
        trial = 0
        while True:
            if trial > 4:
                break
            try:
                normalized = json.loads(response)
                for nid, _ in enumerate(normalized):
                    normalized[nid]["source"] = self.text.strip()
                break
            except Exception as e:
                if not self.args.suppress_error_reports:
                    print("Error:", e)
                    print(f"Index {self.idx}: Retrying the trial {trial}...")
                response = json_correction(response, str(e), self.api_key)
                trial += 1
        if trial < 5:
            return normalized
        else:
            return None
                