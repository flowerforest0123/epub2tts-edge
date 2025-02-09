import json
import ssl
from ssl import SSLContext
from typing import Optional, AsyncGenerator

import aiohttp
import certifi
from edge_tts import Communicate
from edge_tts.communicate import date_to_string, ssml_headers_plus_data, connect_id, mkssml, get_headers_and_data
from edge_tts.constants import DEFAULT_VOICE, WSS_URL, SEC_MS_GEC_VERSION, WSS_HEADERS
from edge_tts.drm import DRM
from edge_tts.exceptions import UnknownResponse, UnexpectedResponse, WebSocketError, NoAudioReceived
from edge_tts.typing import TTSChunk


class ReusableCommunicate(Communicate):
    def __init__(self, text: str, voice: str = DEFAULT_VOICE, ssl_context: SSLContext = ssl.create_default_context(cafile=certifi.where()), *, rate: str = "+0%", volume: str = "+0%",
                 pitch: str = "+0Hz", connector: Optional[aiohttp.BaseConnector] = None, proxy: Optional[str] = None,
                 connect_timeout: Optional[int] = 10, receive_timeout: Optional[int] = 60):
        super().__init__(text, voice, rate=rate, volume=volume, pitch=pitch, connector=connector, proxy=proxy,
                         connect_timeout=connect_timeout, receive_timeout=receive_timeout)
        self.ssl_ctx = ssl_context

    async def __stream(self) -> AsyncGenerator[TTSChunk, None]:
        async def send_command_request() -> None:
            """Sends the command request to the service."""
            await websocket.send_str(
                f"X-Timestamp:{date_to_string()}\r\n"
                "Content-Type:application/json; charset=utf-8\r\n"
                "Path:speech.config\r\n\r\n"
                '{"context":{"synthesis":{"audio":{"metadataoptions":{'
                '"sentenceBoundaryEnabled":"false","wordBoundaryEnabled":"true"},'
                '"outputFormat":"audio-24khz-48kbitrate-mono-mp3"'
                "}}}}\r\n"
            )

        async def send_ssml_request() -> None:
            """Sends the SSML request to the service."""
            await websocket.send_str(
                ssml_headers_plus_data(
                    connect_id(),
                    date_to_string(),
                    mkssml(
                        self.tts_config,
                        self.state["partial_text"],
                    ),
                )
            )

        # audio_was_received indicates whether we have received audio data
        # from the websocket. This is so we can raise an exception if we
        # don't receive any audio data.
        audio_was_received = False

        # Create a new connection to the service.
        #ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        async with aiohttp.ClientSession(
            connector=self.connector,
            trust_env=True,
            timeout=self.session_timeout,
        ) as session, session.ws_connect(
            f"{WSS_URL}&Sec-MS-GEC={DRM.generate_sec_ms_gec()}"
            f"&Sec-MS-GEC-Version={SEC_MS_GEC_VERSION}"
            f"&ConnectionId={connect_id()}",
            compress=15,
            proxy=self.proxy,
            headers=WSS_HEADERS,
            ssl=self.ssl_ctx,
        ) as websocket:
            await send_command_request()

            await send_ssml_request()

            async for received in websocket:
                if received.type == aiohttp.WSMsgType.TEXT:
                    encoded_data: bytes = received.data.encode("utf-8")
                    parameters, data = get_headers_and_data(
                        encoded_data, encoded_data.find(b"\r\n\r\n")
                    )

                    path = parameters.get(b"Path", None)
                    if path == b"audio.metadata":
                        # Parse the metadata and yield it.
                        parsed_metadata = self.__parse_metadata(data)
                        yield parsed_metadata

                        # Update the last duration offset for use by the next SSML request.
                        self.state["last_duration_offset"] = (
                            parsed_metadata["offset"] + parsed_metadata["duration"]
                        )
                    elif path == b"turn.end":
                        # Update the offset compensation for the next SSML request.
                        self.state["offset_compensation"] = self.state[
                            "last_duration_offset"
                        ]

                        # Use average padding typically added by the service
                        # to the end of the audio data. This seems to work pretty
                        # well for now, but we might ultimately need to use a
                        # more sophisticated method like using ffmpeg to get
                        # the actual duration of the audio data.
                        self.state["offset_compensation"] += 8_750_000

                        # Exit the loop so we can send the next SSML request.
                        break
                    elif path not in (b"response", b"turn.start"):
                        raise UnknownResponse("Unknown path received")
                elif received.type == aiohttp.WSMsgType.BINARY:
                    # Message is too short to contain header length.
                    if len(received.data) < 2:
                        raise UnexpectedResponse(
                            "We received a binary message, but it is missing the header length."
                        )

                    # The first two bytes of the binary message contain the header length.
                    header_length = int.from_bytes(received.data[:2], "big")
                    if header_length > len(received.data):
                        raise UnexpectedResponse(
                            "The header length is greater than the length of the data."
                        )

                    # Parse the headers and data from the binary message.
                    parameters, data = get_headers_and_data(
                        received.data, header_length
                    )

                    # Check if the path is audio.
                    if parameters.get(b"Path") != b"audio":
                        raise UnexpectedResponse(
                            "Received binary message, but the path is not audio."
                        )

                    # At termination of the stream, the service sends a binary message
                    # with no Content-Type; this is expected. What is not expected is for
                    # an MPEG audio stream to be sent with no data.
                    content_type = parameters.get(b"Content-Type", None)
                    if content_type not in [b"audio/mpeg", None]:
                        raise UnexpectedResponse(
                            "Received binary message, but with an unexpected Content-Type."
                        )

                    # We only allow no Content-Type if there is no data.
                    if content_type is None:
                        if len(data) == 0:
                            continue

                        # If the data is not empty, then we need to raise an exception.
                        raise UnexpectedResponse(
                            "Received binary message with no Content-Type, but with data."
                        )

                    # If the data is empty now, then we need to raise an exception.
                    if len(data) == 0:
                        raise UnexpectedResponse(
                            "Received binary message, but it is missing the audio data."
                        )

                    # Yield the audio data.
                    audio_was_received = True
                    yield {"type": "audio", "data": data}
                elif received.type == aiohttp.WSMsgType.ERROR:
                    raise WebSocketError(
                        received.data if received.data else "Unknown error"
                    )

            if not audio_was_received:
                raise NoAudioReceived(
                    "No audio was received. Please verify that your parameters are correct."
                )

    async def stream(
        self,
    ) -> AsyncGenerator[TTSChunk, None]:
        """
        Streams audio and metadata from the service.

        Raises:
            NoAudioReceived: If no audio is received from the service.
            UnexpectedResponse: If the response from the service is unexpected.
            UnknownResponse: If the response from the service is unknown.
            WebSocketError: If there is an error with the websocket.
        """

        # Check if stream was called before.
        if self.state["stream_was_called"]:
            raise RuntimeError("stream can only be called once.")
        self.state["stream_was_called"] = True

        # Stream the audio and metadata from the service.
        for self.state["partial_text"] in self.texts:
            try:
                async for message in self.__stream():
                    yield message
            except aiohttp.ClientResponseError as e:
                if e.status != 403:
                    raise

                DRM.handle_client_response_error(e)
                async for message in self.__stream():
                    yield message

    def __parse_metadata(self, data: bytes) -> TTSChunk:
        for meta_obj in json.loads(data)["Metadata"]:
            meta_type = meta_obj["Type"]
            if meta_type == "WordBoundary":
                current_offset = (
                    meta_obj["Data"]["Offset"] + self.state["offset_compensation"]
                )
                current_duration = meta_obj["Data"]["Duration"]
                return {
                    "type": meta_type,
                    "offset": current_offset,
                    "duration": current_duration,
                    "text": meta_obj["Data"]["text"]["Text"],
                }
            if meta_type in ("SessionEnd",):
                continue
            raise UnknownResponse(f"Unknown metadata type: {meta_type}")
        raise UnexpectedResponse("No WordBoundary metadata found")
