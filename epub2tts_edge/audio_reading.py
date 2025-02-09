import asyncio
import hashlib
import io
import os
import re
import ssl
from pathlib import Path

import certifi
import edge_tts
from nltk import sent_tokenize
from pydub import AudioSegment
from tqdm.asyncio import tqdm

import reusable_communicate

AMOUNT_PARALLEL_SENTENCE_TASKS = 10
AMOUNT_PARALLEL_PARAGRAPH_TASKS = 2

MAX_WORDS_FOR_TTS = 100

cache_folder_path = "./cache"


def check_cache_file_exists_and_is_not_empty(filename):
    return (os.path.exists(os.path.join(cache_folder_path, filename))
            and os.path.getsize(os.path.join(cache_folder_path, filename)) > 0)


class Sentence:

    def __init__(self, text, paragraph, audio=None):
        self.text = text
        self.paragraph = paragraph
        self.speaker = paragraph.speaker
        self.audio = audio

    async def process_text_to_audio(self):
        for speakattempt in range(3):
            try:
                await self.stream_tts()
                return  # Falls erfolgreich, direkt rausgehen
            except Exception as e:
                print(f"Attempt {speakattempt + 1}/3 failed with '{self.text}' in run_edgespeak with error: {e}")
                await asyncio.sleep(3)

        print(f"Giving up on sentence '{self.text}' after 3 attempts.")
        self.audio = None

    async def stream_tts(self):
        communicate = reusable_communicate.ReusableCommunicate(self.text, self.speaker, self.paragraph.chapter.book.ssl_context)
        stream = io.BytesIO()

        async for chunk in communicate.stream():
            if isinstance(chunk, dict):
                if chunk.get("type") == "audio":  # Keep only audio chunks
                    stream.write(chunk["data"])  # Extract audio data
            else:
                if isinstance(chunk, bytes):
                    stream.write(chunk)

        stream.seek(0)  # Reset pointer to the beginning
        self.audio = AudioSegment.from_file(stream, format="mp3")

    def clean_up(self):
        self.audio = None


def fix_sentence_text(text):
    new_text = re.sub(r'[!]+', '!', text)
    new_text = re.sub(r'[?]+', '?', new_text)
    return new_text


class Paragraph:

    def __init__(self, text, chapter, sentence_silence_length=1200, sentences=None, audio=None):
        self.speaker = chapter.speaker
        self.text = text
        self.sentence_silence = AudioSegment.silent(sentence_silence_length)
        self.sentences = sentences if sentences is not None else []
        self.audio = audio
        self.chapter = chapter
        self.filename = self.generate_filename()

    def update_text(self, text):
        self.text = text
        self.filename = self.generate_filename()

    def generate_filename(self):
        gen_hash = hashlib.sha256(self.text.encode()).hexdigest()
        return f'{gen_hash}.flac'

    def break_text_into_sentences(self):
        if self.sentences:
            return
        self.sentences = [Sentence(fix_sentence_text(s), self) for s in sent_tokenize(self.text)]

    async def process_text_to_audio(self):
        if check_cache_file_exists_and_is_not_empty(self.filename):
            self.audio = AudioSegment.from_file(os.path.join(cache_folder_path, self.filename))
            return

        if not self.sentences:
            self.break_text_into_sentences()

        semaphore = asyncio.Semaphore(AMOUNT_PARALLEL_SENTENCE_TASKS)

        async def process_sentence(sentence):
            async with semaphore:
                await sentence.process_text_to_audio()

        await asyncio.gather(*(process_sentence(s) for s in self.sentences))

        self.audio = AudioSegment.empty()

        first = True
        for s in self.sentences:
            if s.audio:
                if not first:
                    self.audio += self.sentence_silence
                self.audio += s.audio
                first = False

        self.audio.export(os.path.join(cache_folder_path, self.filename), format='flac')

    def clean_up(self):
        for s in self.sentences:
            s.clean_up()
        self.audio = None
        if check_cache_file_exists_and_is_not_empty(self.filename):
            os.remove(os.path.join(cache_folder_path, self.filename))


class Chapter:

    def __init__(self, paragraphs_as_text, book, title=None, paragraph_silence_length=1200, sentence_silence_length=1200,
                 paragraphs=None, audio=None):
        self.paragraphs_as_text = paragraphs_as_text
        self.book = book
        self.speaker = book.speaker
        self.title = title
        self.paragraph_silence = AudioSegment.silent(paragraph_silence_length)
        self.sentence_silence_length = sentence_silence_length
        self.paragraphs = paragraphs if paragraphs is not None else []
        self.audio = audio
        self.filename = self.generate_filename()

    def generate_filename(self):
        if len(self.paragraphs_as_text) > 0:
            gen_hash = hashlib.sha256("".join(self.paragraphs_as_text).encode()).hexdigest()
            return f'{gen_hash}.flac'
        else:
            if self.title is not None:
                gen_hash = hashlib.sha256(self.title.encode()).hexdigest()
                return f'{gen_hash}.flac'
        return None

    def break_text_into_paragraphs(self):
        if self.paragraphs:
            return
        if self.title is not None and self.title not in ['Title', 'blank']:
            self.paragraphs.append(Paragraph(self.title, self, self.sentence_silence_length))
        for paragraph_as_text in self.paragraphs_as_text:
            if paragraph_as_text:
                self.paragraphs.append(Paragraph(paragraph_as_text, self, self.sentence_silence_length))

    async def process_text_to_audio(self):
        if check_cache_file_exists_and_is_not_empty(self.filename):
            self.audio = AudioSegment.from_file(os.path.join(cache_folder_path, self.filename))
            return
        if not self.paragraphs:
            self.break_text_into_paragraphs()

        semaphore = asyncio.Semaphore(AMOUNT_PARALLEL_PARAGRAPH_TASKS)

        async def process_paragraph(paragraph):
            async with semaphore:
                await paragraph.process_text_to_audio()

        tasks = [process_paragraph(p) for p in self.paragraphs]
        await tqdm.gather(*tasks, desc=f'Process Chapter: {self.title}', unit='pg')

        self.audio = AudioSegment.empty()

        first = True
        for p in self.paragraphs:
            if p.audio:
                if not first:
                    self.audio += self.paragraph_silence
                self.audio += p.audio
                p.clean_up()
                first = False
        self.audio.export(os.path.join(cache_folder_path, self.filename), format='flac')

    def clean_up(self):
        for p in self.paragraphs:
            p.clean_up()
        self.audio = None
        if check_cache_file_exists_and_is_not_empty(self.filename):
            os.remove(os.path.join(cache_folder_path, self.filename))

    def get_file_path(self):
        return os.path.join(cache_folder_path, self.filename)


def build_sentences_from_line(line_striped, new_paragraph):
    def fun(x):
        return any(char.isalnum() for char in x)

    # Filter out empty sentences
    strings = list(filter(fun, sent_tokenize(line_striped)))
    if len(strings) == 0:
        return

    current_combined = strings[0]

    for i in range(1, len(strings)):
        if len(current_combined.split()) > MAX_WORDS_FOR_TTS:
            new_paragraph.sentences.append(Sentence(fix_sentence_text(current_combined), new_paragraph))
            current_combined = strings[i]
        else:
            current_combined = current_combined + " " + strings[i]

    new_paragraph.sentences.append(Sentence(fix_sentence_text(current_combined), new_paragraph))


class Book:

    def __init__(self, filename_text, speaker, paragraph_silence_length=1200, sentence_silence_length=1200):
        Path(cache_folder_path).mkdir(parents=True, exist_ok=True)
        self.filename_text = filename_text
        self.author = "Unknown"
        self.title = self.filename_text
        self.chapters = []
        self.paragraph_silence_length = paragraph_silence_length
        self.sentence_silence_length = sentence_silence_length
        self.speaker = speaker
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())

    def read_text_into_chapters(self):
        with open(self.filename_text, "r", encoding="utf-8") as file:
            current_chapter = None
            i = 0
            for line in file:
                if i < 2:
                    i += 1
                    if line.startswith('Title: '):
                        self.title = line.replace('Title: ', '').strip()
                    elif line.startswith('Author: '):
                        self.author = line.replace('Author: ', '').strip()
                    continue

                line_striped = line.strip()
                if line_striped == "":
                    continue
                if line_striped.startswith("#"):
                    if current_chapter and current_chapter.paragraphs:
                        self.chapters.append(current_chapter)
                    title = line[1:].strip() if any(c.isalnum() for c in line_striped) else "blank"
                    current_chapter = Chapter([], self, title=title,
                                              paragraph_silence_length=self.paragraph_silence_length,
                                              sentence_silence_length=self.sentence_silence_length)
                elif any(c.isalnum() for c in line_striped):
                    new_paragraph = Paragraph(" ", current_chapter,
                                              sentence_silence_length=self.sentence_silence_length)
                    build_sentences_from_line(line_striped, new_paragraph)
                    new_paragraph.update_text(" ".join([s.text for s in new_paragraph.sentences]))
                    current_chapter.paragraphs.append(new_paragraph)
            if current_chapter.paragraphs:
                self.chapters.append(current_chapter)

    def process_chapters(self):
        [asyncio.run(c.process_text_to_audio()) for c in self.chapters]

    def clean_up(self):
        for chapter in self.chapters:
            chapter.clean_up()
