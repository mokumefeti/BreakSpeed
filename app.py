import streamlit as st
import numpy as np
import plotly.graph_objects as go

from scipy.signal import find_peaks
from streamlit_autorefresh import st_autorefresh
from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
)

from math import sqrt
import time

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
    font-size:22px;
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
# session_state
# ==================================================

if "samples" not in st.session_state:
    st.session_state.samples = []

if "post_trigger" not in st.session_state:
    st.session_state.post_trigger = False

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "captured_signal" not in st.session_state:
    st.session_state.captured_signal = None

if "trigger_time" not in st.session_state:
    st.session_state.trigger_time = None

if not isinstance(
    st.session_state.samples,
    list
):
    st.session_state.samples = (
        st.session_state.samples.tolist()
    )

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
# 解析設定
# ==================================================

st.header("解析設定")

threshold_ratio = st.slider(
    "ピーク感度",
    0.05,
    1.00,
    0.50,
    0.01
)

trigger_level = st.slider(
    "ブレイク検出レベル 録音開始トリガー",
    1000,
    20000,
    6000,
    500
)

# record_sec = st.slider(
#     "ブレイク後録音時間",
#     1.0,
#     5.0,
#     3.0,
#     0.5
# )
record_sec = 1.5

# ==================================================
# WebRTC
# ==================================================

ctx = webrtc_streamer(
    key="audio",
    mode=WebRtcMode.SENDONLY,
    media_stream_constraints={
        "video": False,
        "audio": True,
    }
)

# ==================================================
# 音声取得
# ==================================================

level = 0

if ctx.state.playing and ctx.audio_receiver:

    try:

        frames = ctx.audio_receiver.get_frames(
            timeout=1
        )

        all_audio = []

        for frame in frames:

            audio = frame.to_ndarray()

            all_audio.extend(
                audio.flatten()
            )

        if len(all_audio):

            audio = np.array(
                all_audio,
                dtype=np.float32
            )

            level = np.max(
                np.abs(audio)
            )

            if not isinstance(
                st.session_state.samples,
                list
            ):
                st.session_state.samples = list(
                    st.session_state.samples
                )

            st.session_state.samples.extend(
                audio.tolist()
            )

    except:
        pass

    st.write("playing =", ctx.state.playing)

    st.write(
        "audio_receiver =",
        ctx.audio_receiver is not None
    )

    try:

        frames = ctx.audio_receiver.get_frames(
            timeout=1
        )

        st.write(
            "受信フレーム数",
            len(frames)
        )

    except Exception as e:

        st.error(e)

    # ==================================================
    # 待機中は直近3秒だけ保持
    # ==================================================
    if not st.session_state.post_trigger:

        MAX_BUFFER = record_sec * SAMPLERATE

        if len(st.session_state.samples) > MAX_BUFFER:

            st.session_state.samples = (
                st.session_state.samples[-MAX_BUFFER:]
            )

            st.write(
                "samples type =",
                type(st.session_state.samples)
            )

            st.write(
                "sample count =",
                len(st.session_state.samples)
            )

# ==================================================
# 状態表示
# ==================================================

st.header("マイク状態")

c1, c2, c3 = st.columns(3)

with c1:

    st.metric(
        "現在音量",
        int(level)
    )

with c2:

    st.metric(
        "保存サンプル数",
        len(st.session_state.samples)
    )

with c3:

    if level < 1000:

        state = "🎤待機中"

    elif level < trigger_level:

        state = "🔊入力あり"

    else:

        state = "💥ブレイク候補"

    st.metric(
        "状態",
        state
    )

progress_value = int(
    min(level / 20000, 1.0) * 100
)

st.progress(progress_value)

# ==================================================
# リアルタイム波形
# ==================================================

if len(st.session_state.samples):

    wave = np.array(
        st.session_state.samples,
        dtype=np.float32
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            y=wave,
            mode="lines",
            name="Audio"
        )
    )

    fig.update_layout(
        title="リアルタイム波形",
        height=350
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==================================================
# ブレイク検出
# ==================================================

if (
    level > trigger_level
    and
    not st.session_state.post_trigger
    and
    not st.session_state.analysis_done
):

    st.session_state.post_trigger = True

    st.session_state.trigger_time = time.time()

# ==================================================
# ブレイク後録音
# ==================================================

if st.session_state.post_trigger:

    elapsed = (
        time.time()
        - st.session_state.trigger_time
    )

    st.info(
        f"💥 ブレイク検出後録音中 {elapsed:.1f}/{record_sec:.1f} sec"
    )

    if elapsed >= record_sec:

        st.session_state.post_trigger = False

        st.session_state.analysis_done = True

        st.session_state.captured_signal = np.array(
            st.session_state.samples,
            dtype=np.float32
        )

        st.success(
            "✅ 録音完了"
        )

# ==================================================
# 解析
# ==================================================

if (
    st.session_state.analysis_done
    and
    st.session_state.captured_signal is not None
):

    signal = st.session_state.captured_signal.copy()

    peak = np.max(np.abs(signal))

    if peak < 1:

        st.error("音量が小さすぎます")

    else:

        signal /= peak

        absbuf = np.abs(signal)

        threshold = (
            np.max(absbuf)
            * threshold_ratio
        )

        peaks, _ = find_peaks(
            absbuf,
            height=threshold,
            prominence=threshold,
            distance=int(
                0.02 * SAMPLERATE
            )
        )

        st.subheader("解析結果")

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
            height=500
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

        st.write(
            f"ピーク数: {len(peaks)}"
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

                if dt > 0:

                    speed_ms = (
                        distance_cm / 100
                    ) / dt

                    speed_kmh = (
                        speed_ms * 3.6
                    )

                    st.success(
                        f"推定速度 : {speed_kmh:.1f} km/h"
                    )

                    st.write(
                        f"時間差 : {dt*1000:.1f} ms"
                    )

                else:

                    st.error(
                        "ピーク検出異常"
                    )

            else:

                st.warning(
                    "1つ目のピークが見つかりません"
                )

        else:

            st.warning(
                "ピークが2個見つかりません"
            )


# ==================================================
# リセット
# ==================================================

if st.button("次の測定"):

    st.session_state.samples = []

    st.session_state.post_trigger = False

    st.session_state.analysis_done = False

    st.session_state.captured_signal = None

    st.session_state.trigger_time = None

    st.rerun()

