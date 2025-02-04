import asyncio
import hashlib
import io
import os
import re

import edge_tts
from nltk import sent_tokenize
from pydub import AudioSegment
from tqdm.asyncio import tqdm

MAX_WORDS_FOR_TTS = 50

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
        communicate = edge_tts.Communicate(self.text, self.speaker)
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

    def __init__(self, text, chapter, sentence_silence_length=0, sentences=None, audio=None):
        self.speaker = chapter.speaker
        self.text = text
        self.sentence_silence = AudioSegment.silent(sentence_silence_length)
        self.sentences = sentences if sentences is not None else []
        self.audio = audio
        self.chapter = chapter
        self.filename = self.generate_filename()

    def generate_filename(self):
        gen_hash = hashlib.sha256(self.text.encode()).hexdigest()
        return f'{gen_hash}.flac'

    def break_text_into_sentences(self):
        if self.sentences:
            return
        strings = sent_tokenize(self.text)
        last_index = 0
        for i in range(len(strings)):
            combined = ' '.join(strings[last_index:i + 1])
            if len(combined.split()) > MAX_WORDS_FOR_TTS:
                self.sentences.append(Sentence(fix_sentence_text(combined), self))
                last_index = i
        if last_index < len(strings):
            combined = ' '.join(strings[last_index:])
            self.sentences.append(Sentence(fix_sentence_text(combined), self))

    async def process_text_to_audio(self):
        if check_cache_file_exists_and_is_not_empty(self.filename):
            self.audio = AudioSegment.from_file(os.path.join(cache_folder_path, self.filename))
            return

        if not self.sentences:
            self.break_text_into_sentences()

        semaphore = asyncio.Semaphore(10)

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

    def __init__(self, paragraphs_as_text, speaker, title=None, paragraph_silence_length=0, sentence_silence_length=0,
                 paragraphs=None, audio=None):
        self.paragraphs_as_text = paragraphs_as_text
        self.speaker = speaker
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

        semaphore = asyncio.Semaphore(2)

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
