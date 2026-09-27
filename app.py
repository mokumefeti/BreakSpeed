import streamlit as st
import numpy as np
import plotly.graph_objects as go
import threading
import time

from math import sqrt
from scipy.signal import find_peaks
from streamlit_autorefresh import st_autorefresh

from streamlit_webrtc import (
    webrtc_streamer,
    AudioProcessorBase,
)

# ==================================================
# 定数
# ==================================================

BALL_DIA = 5.71
SAMPLERATE = 48000

# ==================================================
# ページ設定
# ==================================================

st.set_page_config(
    page_title="BreakSpeed",
    layout="wide"
)

st_autorefresh(
    interval=500,
    key="refresh"
)

st.markdown("""
<style>

html, body, [class*="css"] {
    font-size: 22px;
}

p, label {
    font-size:22px !important;
}

[data-testid="metric-container"]{
    font-size:28px !important;
}

</style>
""", unsafe_allow_html=True)

st.title("🎱 BreakSpeed")

# ==================================================
# 距離設定
# ==================================================

st.header("手玉位置")

col1, col2 = st.columns(2)

with col1:
    x2 = st.number_input(
        "ヘッドから球何個分後ろ",
        min_value=0.0,
        value=0.0,
        step=0.1
    )

with col2:
    y2 = st.number_input(
        "レールから球何個分離す",
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

# ==================================================
# 感度
# ==================================================

st.header("解析設定")

threshold_ratio = st.slider(
    "ピーク感度",
    0.05,
    0.50,
    0.15,
    0.01
)

trigger_level = st.slider(
    "録音トリガ",
    0.01,
    0.80,
    0.15,
    0.01
)

record_sec = st.slider(
    "録音時間",
    0.5,
    2.0,
    1.0,
    0.1
)

# ==================================================
# Audio
# ==================================================

class AudioProcessor(AudioProcessorBase):

    def __init__(self):

        self.lock = threading.Lock()

        self.samples = []

        self.level = 0

        self.state = "waiting"

        self.detected = False

        self.detect_time = None

        self.analysis_done = False

    def recv(self, frame):

        audio = frame.to_ndarray()

        audio = audio.flatten().astype(np.float32)

        with self.lock:

            self.samples.extend(audio.tolist())

            if len(self.samples) > SAMPLERATE * 5:
                self.samples = self.samples[-SAMPLERATE * 5:]

            level = np.max(np.abs(audio))

            self.level = level / 32768.0

            if (
                not self.detected
                and
                self.level > trigger_level
            ):
                self.detected = True
                self.detect_time = time.time()
                self.state = "recording"

        return frame

# ==================================================
# WebRTC
# ==================================================

ctx = webrtc_streamer(
    key="breakspeed",
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={
        "video": False,
        "audio": True
    }
)

# ==================================================
# 状態表示
# ==================================================

if ctx.audio_processor:

    proc = ctx.audio_processor

    st.metric(
        "マイクレベル",
        f"{proc.level:.3f}"
    )

    if proc.state == "waiting":

        st.info(
            "🎤 待機中... ブレイクしてください"
        )

    elif proc.state == "recording":

        elapsed = (
            time.time()
            - proc.detect_time
        )

        st.success(
            f"💥 衝突音検出 {elapsed:.1f} sec"
        )

    # ==================================================
    # 波形表示
    # ==================================================

    with proc.lock:

        recent = np.array(
            proc.samples,
            dtype=np.float32
        )

    if len(recent) > 1000:

        display_count = min(
            len(recent),
            int(display_sec * SAMPLERATE)
        )

        wave = recent[-display_count:]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                y=wave,
                mode="lines",
                name="Wave"
            )
        )

        fig.update_layout(
            height=300,
            title="音声波形"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ==================================================
    # 自動解析
    # ==================================================

    if (
        proc.detected
        and
        not proc.analysis_done
    ):

        elapsed = (
            time.time()
            - proc.detect_time
        )

        if elapsed >= record_sec:

            proc.analysis_done = True

            with proc.lock:

                signal = np.array(
                    proc.samples,
                    dtype=np.float32
                )

            peak = np.max(
                np.abs(signal)
            )

            if peak < 10:

                st.error(
                    "音量が小さすぎます"
                )

            else:

                signal = signal / peak

                absbuf = np.abs(signal)

                threshold = (
                    np.max(absbuf)
                    * threshold_ratio
                )

                peaks, prop = find_peaks(
                    absbuf,
                    height=threshold,
                    prominence=threshold,
                    distance=int(
                        0.02 * SAMPLERATE
                    )
                )

                fig2 = go.Figure()

                fig2.add_trace(
                    go.Scatter(
                        y=signal,
                        mode="lines",
                        name="Signal"
                    )
                )

                if len(peaks):

                    fig2.add_trace(
                        go.Scatter(
                            x=peaks,
                            y=signal[peaks],
                            mode="markers",
                            marker=dict(
                                size=10,
                                color="red"
                            ),
                            name="Peaks"
                        )
                    )

                fig2.update_layout(
                    height=400,
                    title="解析結果"
                )

                st.plotly_chart(
                    fig2,
                    use_container_width=True
                )

                st.write(
                    f"検出ピーク数: {len(peaks)}"
                )

                if len(peaks) >= 2:

                    peak2 = peaks[
                        np.argmax(
                            absbuf[peaks]
                        )
                    ]

                    before = peaks[
                        peaks < peak2
                    ]

                    if len(before):

                        peak1 = before[-1]

                        dt = (
                            peak2 - peak1
                        ) / SAMPLERATE

                        speed = (
                            distance_cm / dt
                        ) / 100

                        speed_kmh = (
                            speed * 3.6
                        )

                        st.success(
                            f"推定速度 : {speed_kmh:.1f} km/h"
                        )

                        st.write(
                            f"時間差 : {dt*1000:.1f} ms"
                        )

                    else:

                        st.warning(
                            "1つ目のピークが見つかりません"
                        )

                else:

                    st.warning(
                        "ピークが2個見つかりません"
                    )