import os
import json
import openai
import logging
import backoff
from pathlib import Path
from textwrap import dedent
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

def save_user_input(story_path, old_name, new_name, background, cultural_elements):
    input_data = {
        "timestamp": datetime.now().isoformat(),
        "story_path": story_path,
        "old_name": old_name,
        "new_name": new_name,
        "background": background,
        "cultural_elements": cultural_elements
    }
    
    with open("story_user_input.json", "a") as f:
        json.dump(input_data, f)
        f.write("\n")

def chunk_story(story, max_words=4000):
    paragraphs = story.split("\n\n")
    chunks = []
    current_chunk = []
    current_word_count = 0

    for paragraph in paragraphs:
        paragraph_word_count = len(paragraph.split())
        if current_word_count + paragraph_word_count > max_words:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [paragraph]
            current_word_count = paragraph_word_count
        else:
            current_chunk.append(paragraph)
            current_word_count += paragraph_word_count

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    return chunks

def get_ideal_prompt(old_name, new_name, story_name, background, cultural_elements):
    return dedent(f"""
        You are a creative writer tasked with culturally adapting and rewriting the story "{story_name}".
        Replace the original protagonist "{old_name}" with "{new_name}", who has the following background: {background}.
        
        Cultural adaptation elements:
        {cultural_elements}
        
        Instructions:
        1. Replace cultural elements, festivals, traditions, and mythological references with appropriate equivalents
        2. Adapt the setting and environment to match the new cultural context
        3. Transform supporting characters to fit the new cultural setting
        4. Maintain the core moral/message while adapting it to the new cultural framework
        
        Make this a complete cultural adaptation while preserving the story's emotional impact and core themes.
    """)

@backoff.on_exception(
    backoff.expo,
    (openai.RateLimitError, openai.APIError),
    max_tries=8,
    max_time=300
)
def rewrite_chunk_with_ai(chunk, prompt):
    try:
        response = openai.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are an expert in cultural adaptation and creative writing."},
                {"role": "user", "content": f"{prompt}\n\nOriginal text:\n{chunk}"}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except openai.BadRequestError as e:
        if "maximum context length" in str(e):
            logger.warning("Token limit exceeded. Breaking into smaller chunks...")
            smaller_chunks = chunk_story(chunk, max_words=2000)
            rewritten_parts = [rewrite_chunk_with_ai(c, prompt) for c in smaller_chunks]
            return "\n\n".join(rewritten_parts)
        raise

def process_story(file_path, old_name, new_name, background, cultural_elements):
    if not all(x.strip() for x in [old_name, new_name, background, cultural_elements]):
        raise ValueError("All fields must be non-empty")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Story file not found: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
    except Exception as e:
        raise RuntimeError(f"Error reading the story file: {e}")

    story_name = Path(file_path).stem
    prompt = get_ideal_prompt(old_name, new_name, story_name, background, cultural_elements)
    chunks = chunk_story(text)
    
    rewritten_chunks = []
    for chunk in chunks:
        try:
            rewritten = rewrite_chunk_with_ai(chunk, prompt)
            rewritten_chunks.append(rewritten)
        except Exception as e:
            logger.error(f"Error processing chunk: {e}")
            rewritten_chunks.append(chunk)
    
    return "\n\n".join(rewritten_chunks)

def main():
    logger.info("Welcome to the Cultural Story Adaptation Tool!")
    story_path = input("Enter the path to the story file: ").strip()
    if not os.path.exists(story_path):
        logger.error("Error: File not found. Please enter a valid file path.")
        return

    old_name = input("Enter the name of the original protagonist: ").strip()
    new_name = input("Enter the new name of the protagonist: ").strip()
    background = input(f"Describe the background for {new_name}: ").strip()
    
    print("\nSpecify cultural elements to adapt (e.g., 'Replace Christmas with Diwali,")
    print("replace ghosts with atmas, adapt Victorian London to modern Mumbai')")
    cultural_elements = input("Cultural adaptations: ").strip()

    save_user_input(story_path, old_name, new_name, background, cultural_elements)
    
    logger.info("Processing your story... This might take a few minutes.")
    try:
        rewritten_story = process_story(story_path, old_name, new_name, background, cultural_elements)
        output_path = Path(story_path).with_suffix(".adapted.txt")
        with open(output_path, "w", encoding="utf-8") as output_file:
            output_file.write(rewritten_story)
        logger.info(f"Adapted story saved to: {output_path}")
    except Exception as e:
        logger.error(f"Error processing the story: {e}")

if __name__ == "__main__":
    main()