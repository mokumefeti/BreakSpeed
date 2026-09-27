import streamlit as st
import numpy as np
import threading

from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    AudioProcessorBase
)

st.set_page_config(layout="wide")

st.title("マイクテスト")

class AudioProcessor(AudioProcessorBase):

    def __init__(self):

        self.lock = threading.Lock()

        self.level = 0
        self.max_level = 0

        self.packet_count = 0

    def recv(self, frame):

        audio = frame.to_ndarray()

        audio = audio.astype(np.float32)

        level = np.max(np.abs(audio))

        with self.lock:

            self.packet_count += 1

            self.level = float(level)

            if level > self.max_level:
                self.max_level = float(level)

        return frame


# ctx = webrtc_streamer(
#     key="test",
#     audio_processor_factory=AudioProcessor,
#     media_stream_constraints={
#         "audio": True,
#         "video": False,
#     }
# )

ctx = webrtc_streamer(
    key="audio",
    mode=WebRtcMode.SENDONLY,
    rtc_configuration={
        "iceServers": [
            {
                "urls": [
                    "stun:stun.l.google.com:19302"
                ]
            }
        ]
    },
    media_stream_constraints={
        "video": False,
        "audio": True
    },
    audio_processor_factory=AudioProcessor,
)

st.write("processor =", ctx.audio_processor)
st.write("playing =", ctx.state.playing)

if ctx.audio_processor:

    proc = ctx.audio_processor

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "現在音量",
            f"{proc.level:.0f}"
        )

    with c2:
        st.metric(
            "最大音量",
            f"{proc.max_level:.0f}"
        )

    with c3:
        st.metric(
            "受信パケット",
            proc.packet_count
        )

    st.progress(
        min(proc.level / 20000, 1.0)
    )