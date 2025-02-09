import asyncio
import hashlib
import io
import os
import re
import ssl
from pathlib import Path

import edge_tts
from nltk import sent_tokenize
from pydub import AudioSegment
from tqdm.asyncio import tqdm

AMOUNT_PARALLEL_PARAGRAPH_TASKS = 10

cache_folder_path = "./cache"


def check_cache_file_exists_and_is_not_empty(filename):
    return (os.path.exists(os.path.join(cache_folder_path, filename))
            and os.path.getsize(os.path.join(cache_folder_path, filename)) > 0)


class Sentence:

    def __init__(self, text, paragraph):
        self.text = text
        self.paragraph = paragraph


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

        audio, word_boundaries = await self.stream_tts()

        self.audio = self.cut_audio(audio, word_boundaries)

        self.audio.export(os.path.join(cache_folder_path, self.filename), format='flac')

    async def stream_tts(self):
        communicate = edge_tts.Communicate(self.text, self.speaker)
        stream = io.BytesIO()
        word_boundaries = []

        async for chunk in communicate.stream():
            if isinstance(chunk, dict) and chunk.get("type") == "WordBoundary":
                word_boundaries.append(chunk)
            elif isinstance(chunk, bytes) or (isinstance(chunk, dict) and chunk.get("type") == "audio"):
                stream.write(chunk.get("data", b""))  # Avoids errors if "data" key is missing

        stream.seek(0)  # Reset pointer to the beginning
        return AudioSegment.from_file(stream, format="mp3"), word_boundaries

    def cut_audio(self, audio, words):
        if not self.sentences or not words:
            return audio  # Handle empty input safely

        indices = []
        index = 0

        def contains_alnum(x):
            return any(char.isalnum() for char in x)

        for sentence in self.sentences:
            words_in_sentence = list(filter(contains_alnum, sentence.text.split()))
            amount_words = len(words_in_sentence)
            index += amount_words - 1
            indices.append(index)
            index += 1  # Account for spacing

        if len(indices) <= 1:
            return audio  # No need to split if there's only one sentence or none

        indices.pop(-1)  # Remove last index to avoid out-of-bounds errors

        split_ranges = [(0, round((words[indices[0]]['offset'] + words[indices[0]]['duration'])/10000))]

        for j in range(1, len(indices)):
            if indices[j - 1] + 1 >= len(words) or indices[j] >= len(words):
                continue  # Avoid accessing invalid indexes

            offset_start = words[indices[j - 1] + 1]['offset']
            sentence_end = round((words[indices[j]]['offset']  + words[indices[j]]['duration'])/10000)
            split_ranges.append((offset_start, sentence_end))

        split_ranges.append((round(words[indices[-1]+1]['offset']/10000), -1))

        new_audio = AudioSegment.empty()
        silence = self.sentence_silence

        for i, (x, y) in enumerate(split_ranges):
            new_audio += audio[x:y]
            if i < len(split_ranges) - 1:
                new_audio += silence

        return new_audio

    def clean_up(self):
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

    for s in strings:
        new_paragraph.sentences.append(Sentence(fix_sentence_text(s), new_paragraph))


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

    def read_text_into_chapters(self):
        with open(self.filename_text, "r", encoding="utf-8") as file:
            current_chapter = None
            i = 0
            for line in file:
                if i < 2:
                    i += 1
                    #TODO Ignore condition if first line didnt contain
                    if line.startswith('Title: '):
                        self.title = line.replace('Title: ', '').strip()
                        continue
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
