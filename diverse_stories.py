import os
import openai
import logging
import backoff
from pathlib import Path
from textwrap import dedent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

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

def get_ideal_prompt(old_name, new_name, story_name, background):
    return dedent(f"""
        You are a creative writer tasked with rewriting the story "{story_name}".
        Replace the original protagonist "{old_name}" with a new character named "{new_name}", who has the following background: {background}.
        Ensure to reflect their unique personality, cultural nuances, and perspective while retaining the original story's core plot.
        Make the character seamlessly fit into the story, preserving its fantastical or real-world elements.
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
                {"role": "system", "content": "You are an assistant helping rewrite stories for diverse characters."},
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

def process_story(file_path, old_name, new_name, background):
    if not old_name.strip() or not new_name.strip() or not background.strip():
        raise ValueError("Names and background cannot be empty")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Story file not found: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
    except Exception as e:
        raise RuntimeError(f"Error reading the story file: {e}")

    story_name = Path(file_path).stem
    prompt = get_ideal_prompt(old_name, new_name, story_name, background)
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
    logger.info("Welcome to the Story Rewriting Tool!")
    story_path = input("Enter the path to the story file: ").strip()
    if not os.path.exists(story_path):
        logger.error("Error: File not found. Please enter a valid file path.")
        return

    old_name = input("Enter the name of the original protagonist: ").strip()
    new_name = input("Enter the new name of the protagonist: ").strip()
    background = input(f"Describe the background for {new_name}: ").strip()

    logger.info("Processing your story... This might take a few minutes.")
    try:
        rewritten_story = process_story(story_path, old_name, new_name, background)
        output_path = Path(story_path).with_suffix(".rewritten.txt")
        with open(output_path, "w", encoding="utf-8") as output_file:
            output_file.write(rewritten_story)
        logger.info(f"Rewritten story saved to: {output_path}")
    except Exception as e:
        logger.error(f"Error processing the story: {e}")

if __name__ == "__main__":
    main()