import streamlit as st
import numpy as np
import plotly.graph_objects as go
import threading
import time

from math import sqrt
from scipy.signal import find_peaks
from streamlit_webrtc import (
    webrtc_streamer,
    AudioProcessorBase
)

# =====================================
# 定数
# =====================================

BALL_DIA = 5.71
SAMPLERATE = 48000

# =====================================
# ページ
# =====================================

st.set_page_config(
    page_title="BreakSpeed",
    layout="wide"
)

st.title("🎱 BreakSpeed")

# =====================================
# 距離設定
# =====================================

st.header("手玉位置")

col1, col2 = st.columns(2)

with col1:
    x2 = st.number_input(
        "ヘッドから球（個分後ろ）",
        min_value=0.0,
        value=0.0,
        step=0.1
    )

with col2:
    y2 = st.number_input(
        "レールから球（個分離す）",
        min_value=0.0,
        value=0.0,
        step=0.1
    )

real_x2 = 127 + x2 * BALL_DIA
real_y2 = 63.5 - (y2 + 0.5) * BALL_DIA

distance_cm = (
    sqrt(real_x2**2 + real_y2**2)
    - BALL_DIA
)

st.metric(
    "移動距離",
    f"{distance_cm:.2f} cm"
)

# =====================================
# 感度
# =====================================

threshold_ratio = st.slider(
    "解析感度",
    min_value=0.05,
    max_value=0.50,
    value=0.15,
    step=0.01
)

trigger_level = st.slider(
    "録音トリガ",
    min_value=0.05,
    max_value=0.80,
    value=0.20,
    step=0.01
)

display_sec = st.slider(
    "表示時間",
    min_value=0.1,
    max_value=1.0,
    value=0.8,
    step=0.1
)

# =====================================
# 音声処理
# =====================================

class AudioProcessor(AudioProcessorBase):

    def __init__(self):

        self.lock = threading.Lock()

        self.samples = []

        self.detected = False

        self.detect_time = None

        self.analysis_done = False

    def recv(self, frame):

        audio = frame.to_ndarray()

        audio = audio.flatten().astype(np.float32)

        with self.lock:

            self.samples.extend(audio)

            level = np.max(np.abs(audio))

            if level > 0:

                level_norm = level / 32768.0

                if (
                    not self.detected
                    and
                    level_norm > trigger_level
                ):
                    self.detected = True
                    self.detect_time = time.time()

        return frame

# =====================================
# 録音
# =====================================

ctx = webrtc_streamer(
    key="breakspeed",
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={
        "video": False,
        "audio": True
    }
)

# =====================================
# 状態表示
# =====================================

if ctx.audio_processor:

    proc = ctx.audio_processor

    if not proc.detected:

        st.info(
            "🎤 待機中... ブレイクしてください"
        )

    else:

        elapsed = (
            time.time()
            - proc.detect_time
        )

        st.success(
            "💥 衝突音検出"
        )

        st.write(
            f"録音継続中 {elapsed:.1f} sec"
        )

        # 0.5秒収録後に解析

        if (
            elapsed > 0.5
            and
            not proc.analysis_done
        ):

            proc.analysis_done = True

            signal = np.array(
                proc.samples,
                dtype=np.float32
            )

            if len(signal) < 1000:

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
                    0.08 * SAMPLERATE
                ),
                prominence=threshold
            )

            peak1 = None
            peak2 = None

            if len(peaks) >= 2:

                peak2 = peaks[
                    np.argmax(
                        absbuf[peaks]
                    )
                ]

                candidates = peaks[
                    peaks <
                    peak2 -
                    int(
                        0.08 *
                        SAMPLERATE
                    )
                ]

                if len(candidates):

                    peak1 = candidates[-1]

                    dt = (
                        peak2
                        - peak1
                    )