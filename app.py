import streamlit as st
import av
import numpy as np
import plotly.graph_objects as go

from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode
)

st.title("マイク確認")

ctx = webrtc_streamer(
    key="audio",
    mode=WebRtcMode.SENDONLY,
    media_stream_constraints={
        "video": False,
        "audio": True,
    }
)

if ctx.state.playing:

    st.success("マイク接続成功")

    if ctx.audio_receiver:

        try:

            frames = ctx.audio_receiver.get_frames(
                timeout=1
            )

            st.write(
                "受信フレーム数",
                len(frames)
            )

            audio = []

            for frame in frames:

                arr = frame.to_ndarray()

                audio.extend(
                    arr.flatten()
                )

            audio = np.array(audio)

            if len(audio):

                level = np.max(
                    np.abs(audio)
                )

                st.metric(
                    "音量",
                    int(level)
                )

                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        y=audio[-5000:]
                    )
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        except Exception as e:

            st.error(e)
            