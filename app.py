import streamlit as st
import numpy as np
import wave
from math import sqrt
import matplotlib.pyplot as plt

SAMPLERATE = 48000
BALL_DIA = 5.71

st.set_page_config(
    page_title="BreakSpeed",
    layout="wide"
)

st.title("🎱 BreakSpeed")

# ----------------------------
# 距離設定
# ----------------------------

col1, col2 = st.columns(2)

with col1:
    x2 = st.number_input(
        "ヘッドから球（個分後ろ）",
        min_value=0.0,
        step=0.1,
        value=0.0
    )

with col2:
    y2 = st.number_input(
        "レールから球（個分離す）",
        min_value=0.0,
        step=0.1,
        value=0.0
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

st.divider()

uploaded_file = st.file_uploader(
    "衝突音 wav ファイル",
    type=["wav"]
)

# ----------------------------
# ピーク検出
# ----------------------------

def simple_find_peaks(
    data,
    threshold,
    distance
):
    peaks = []

    last_peak = -distance

    for i in range(1, len(data)-1):

        if (
            data[i] > threshold
            and data[i] > data[i-1]
            and data[i] > data[i+1]
            and i - last_peak >= distance
        ):
            peaks.append(i)
            last_peak = i

    return np.array(peaks)

# ----------------------------
# メイン解析
# ----------------------------

if uploaded_file:

    wav = wave.open(uploaded_file)

    sample_rate = wav.getframerate()

    frames = wav.readframes(
        wav.getnframes()
    )

    wav.close()

    audio = np.frombuffer(
        frames,
        dtype=np.int16
    )

    audio = (
        audio.astype(np.float32)
        / 32768.0
    )

    absbuf = np.abs(audio)

    fig, ax = plt.subplots(
        figsize=(12, 4)
    )

    ax.plot(audio)

    ax.set_title("波形")

    st.pyplot(fig)

    peaks = simple_find_peaks(
        absbuf,
        np.max(absbuf) * 0.15,
        int(0.08 * sample_rate)
    )

    if len(peaks) < 2:

        st.error(
            "ピークが2個以上見つかりません"
        )

    else:

        peak2 = peaks[
            np.argmax(absbuf[peaks])
        ]

        candidates = peaks[
            peaks < peak2 - int(0.08 * sample_rate)
        ]

        if len(candidates) == 0:

            st.error(
                "1回目ピーク検出失敗"
            )

        else:

            peak1 = candidates[-1]

            dt = (
                peak2 - peak1
            ) / sample_rate

            speed_kmh = (
                distance_cm / 100
            ) / dt * 3.6

            st.success(
                f"速度 = {speed_kmh:.2f} km/h"
            )

            st.write(
                f"Δt = {dt*1000:.1f} ms"
            )

            fig2, ax2 = plt.subplots(
                figsize=(12, 4)
            )

            ax2.plot(audio)

            ax2.axvline(
                peak1,
                color="yellow",
                label="1回目"
            )

            ax2.axvline(
                peak2,
                color="cyan",
                label="2回目"
            )

            ax2.legend()

            st.pyplot(fig2)