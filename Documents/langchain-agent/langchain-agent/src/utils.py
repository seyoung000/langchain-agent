from openai import OpenAI
import re
from langdetect import detect

def translate_text(text, target_lang='en'):
    """Translate text to the target language using OpenAI."""
    client = OpenAI()
    prompt = f"Translate the following text to {target_lang}: {text}"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


def preprocess_text(query_text):
    # translate text to English if necessary

    cleaned = re.sub(r"[a-fA-F0-9\-]{8,}", "", query_text)  # UUID 제거
    cleaned = re.sub(r"[^가-힣a-zA-Z\s]", "", cleaned)  # 특수문자 제거

    lang = detect(cleaned)

    if lang != 'en':
        translated = translate_text(query_text, target_lang='en')
    return translated, lang