import streamlit as st
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase

st.title("BreakSpeed")

class AudioProcessor(AudioProcessorBase):

    def __init__(self):
        self.samples = []

    def recv(self, frame):

        audio = frame.to_ndarray()

        self.samples.extend(
            audio.flatten().tolist()
        )

        return frame

ctx = webrtc_streamer(
    key="audio",
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={
        "video": False,
        "audio": True
    }
)

if ctx.audio_processor:

    st.write(
        "サンプル数:",
        len(ctx.audio_processor.samples)
    )