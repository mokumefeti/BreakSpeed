import streamlit as st
import numpy as np
import av

from math import sqrt
from scipy.signal import find_peaks
from streamlit_webrtc import (
    webrtc_streamer,
    AudioProcessorBase
)

BALL_DIA = 5.71

st.set_page_config(
    page_title="BreakSpeed",
    layout="wide"
)

st.title("🎱 BreakSpeed")

class AudioProcessor(AudioProcessorBase):

    def __init__(self):
        self.samples = []

    def recv(self, frame):

        audio = frame.to_ndarray()

        self.samples.extend(
            audio.flatten()
        )

        return frame

ctx = webrtc_streamer(
    key="breakspeed",
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={
        "video": False,
        "audio": True
    }
)

x2 = st.number_input(
    "ヘッドから球",
    value=0.0
)

y2 = st.number_input(
    "レールから球",
    value=0.0
)

real_x2 = 127 + x2 * BALL_DIA
real_y2 = 63.5 - (y2 + 0.5) * BALL_DIA

distance_cm = (
    sqrt(real_x2**2 + real_y2**2)
    - BALL_DIA
)

st.metric(
    "距離",
    f"{distance_cm:.2f} cm"
)

threshold_ratio = st.slider(
    "感度",
    0.05,
    0.60,
    0.15
)

if st.button("解析"):

    if not ctx.audio_processor:
        st.stop()

    signal = np.array(
        ctx.audio_processor.samples,
        dtype=np.float32
    )

    samplerate = 48000

    if signal.size < 1000:
        st.error("録音不足")
        st.stop()

    signal /= np.max(
        np.abs(signal)
    )

    absbuf = np.abs(signal)

    threshold = (
        np.max(absbuf)
        * threshold_ratio
    )

    peaks, _ = find_peaks(
        absbuf,
        height=threshold,
        distance=int(
            0.08 * samplerate
        ),
        prominence=threshold
    )

    if len(peaks) < 2:

        st.error(
            f"ピーク不足: {len(peaks)}"
        )

    else:

        peak2 = peaks[
            np.argmax(absbuf[peaks])
        ]

        candidates = peaks[
            peaks <
            peak2 - int(
                0.08 * samplerate
            )
        ]

        if len(candidates) == 0:
            st.error("1発目不明")
            st.stop()

        peak1 = candidates[-1]

        dt = (
            peak2 - peak1
        ) / samplerate

        speed_kmh = (
            distance_cm / 100
        ) / dt * 3.6

        st.success(
            f"{speed_kmh:.2f} km/h"
        )

        st.info(
            f"Δt={dt*1000:.1f}ms"
        )