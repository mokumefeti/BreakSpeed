import streamlit as st
import numpy as np
import soundfile as sf
import plotly.graph_objects as go

from streamlit_mic_recorder import mic_recorder
from scipy.signal import find_peaks
from math import sqrt

# -----------------------------
# 定数
# -----------------------------

BALL_DIA = 5.71

st.set_page_config(
    page_title="BreakSpeed",
    layout="wide"
)

st.title("🎱 BreakSpeed")

# -----------------------------
# 距離設定
# -----------------------------

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

# -----------------------------
# 感度
# -----------------------------

threshold_ratio = st.slider(
    "感度",
    min_value=0.05,
    max_value=0.60,
    value=0.15,
    step=0.01
)

# -----------------------------
# 表示時間
# -----------------------------

display_sec = st.slider(
    "表示時間",
    min_value=0.1,
    max_value=1.0,
    value=0.8,
    step=0.1
)

st.divider()

# -----------------------------
# 録音
# -----------------------------

audio = mic_recorder(
    start_prompt="🎤 録音開始",
    stop_prompt="■ 録音停止",
    just_once=False
)

# -----------------------------
# 解析
# -----------------------------

if audio:

    st.success("録音完了")

    filename = "record.wav"

    with open(filename, "wb") as f:
        f.write(audio["bytes"])

    signal, samplerate = sf.read(filename)

    if len(signal.shape) > 1:
        signal = signal[:, 0]

    signal = signal.astype(np.float32)

    absbuf = np.abs(signal)

    threshold = np.max(absbuf) * threshold_ratio

    peaks, info = find_peaks(
        absbuf,
        height=threshold,
        distance=int(0.08 * samplerate),
        prominence=threshold
    )

    peak1 = None
    peak2 = None

    if len(peaks) >= 2:

        peak2 = peaks[
            np.argmax(absbuf[peaks])
        ]

        candidates = peaks[
            peaks <
            peak2 - int(0.08 * samplerate)
        ]

        if len(candidates) > 0:

            peak1 = candidates[-1]

            dt = (
                peak2 - peak1
            ) / samplerate

            speed_kmh = (
                distance_cm / 100
            ) / dt * 3.6

            st.success(
                f"速度 = {speed_kmh:.2f} km/h"
            )

            st.info(
                f"Δt = {dt*1000:.1f} ms"
            )

    else:
        st.error("ピーク不足")

    # -----------------------------
    # 表示範囲切り出し
    # ----------------*------------

    if peak1 is not None:
        center = peak1
        half_width = int(
            samplerate *
            display_sec/ 2
        )

        start = max(
            0,
            center*- half_width
        )
        end = min(
            len(signal),
           center + half_width
        )

    else:

        start = 0
        end = min(
            len(signal),
            int(
               display_sec * samplerate
            )
        )

    view = signal[start:end]
    t = (
        np.arange(len(view))
        / samplerate
    )

    fig = go.Figure()

    # 波形

    fig.add_trace(
        go.Scatter(
            x=t,
            y=view,
            mode="lines",
            name="wavefo*m",
            line=dict(
                color="lime",
                width=1
            )
        )
    )

    # 閾値線
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="red"
    )

    fig.add_hline(
        y=-threshold,
        line_dash="dash",
        line_color="red"
    )

    # ピーク線

    if peak1 is not None:

        x1 = (
            peak1 - start
        ) / samplerate

        fig.add_vline(
            x=x1,
            line_color="yellow",
            line_width=2
        )

    if peak2 is not None:

        x2line = (
            peak2 - start
        ) / samplerate

        fig.add_vline(
            x=x2line,
            line_color="cyan",
            line_width=2
        )

    fig.update_layout(
        height=500,
        title="波形解析",
        xaxis_title="時間 (sec)",
        yaxis_title="振幅",
        dragmode="zoom",
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.audio(audio["bytes"])