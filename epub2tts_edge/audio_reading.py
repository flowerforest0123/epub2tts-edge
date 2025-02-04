import asyncio
import io
import re

import edge_tts
from nltk import sent_tokenize
from pydub import AudioSegment
from tqdm import tqdm


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
        self.audio = None  # Damit `sum()` später keinen Fehler wirft

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

    def break_text_into_sentences(self):
        if self.sentences:
            return
        self.sentences = [Sentence(fix_sentence_text(s), self) for s in sent_tokenize(self.text)]

    async def process_text_to_audio(self):
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

class Chapter:

    def __init__(self, paragraphs_as_text, speaker, title = None, paragraph_silence_length=0, sentence_silence_length=0, paragraphs=None, audio=None):
        self.paragraphs_as_text = paragraphs_as_text
        self.speaker = speaker
        self.title = title
        self.paragraph_silence = AudioSegment.silent(paragraph_silence_length)
        self.sentence_silence_length = sentence_silence_length
        self.paragraphs = paragraphs if paragraphs is not None else []
        self.audio = audio

    def break_text_into_paragraphs(self):
        if self.paragraphs:
            return
        if self.title is not None and self.title not in ['Title', 'blank']:
            self.paragraphs.append(Paragraph(self.title, self, self.sentence_silence_length))
        for paragraph_as_text in self.paragraphs_as_text:
            self.paragraphs.append(Paragraph(paragraph_as_text, self, self.sentence_silence_length))

    async def process_text_to_audio(self):
        if not self.paragraphs:
            self.break_text_into_paragraphs()

        semaphore = asyncio.Semaphore(2)

        async def process_paragraph(paragraph):
            async with semaphore:
                await paragraph.process_text_to_audio()

        await asyncio.gather(*(process_paragraph(p) for _, p in enumerate(tqdm(self.paragraphs, desc= f'Process Chapter: {self.title}', unit='pg'))))

        self.audio = AudioSegment.empty()

        first = True
        for p in self.paragraphs:
            if p.audio:
                if not first:
                    self.audio += self.paragraph_silence
                self.audio += p.audio
                first = False
