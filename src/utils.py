from openai import OpenAI
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
    client = OpenAI()

    lang = detect(query_text)
    if lang != 'en':
        translated = translate_text(query_text, target_lang='en')
    return translated, lang