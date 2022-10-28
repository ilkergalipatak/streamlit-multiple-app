# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 19:24:46 2022

@author: pc
"""
import asyncio
import logging
import queue
import threading
import urllib.request
from pathlib import Path
from typing import List, NamedTuple, Optional

import av
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydub
import streamlit as st
from aiortc.contrib.media import MediaPlayer
from streamlit_webrtc import (
    RTCConfiguration, WebRtcMode, WebRtcStreamerContext, webrtc_streamer)

HERE = Path(__file__).parent
logger = logging.getLogger(__name__)


def download_file(url, download_to: Path, expected_size=None):
    if download_to.exists():
        if expected_size:
            if download_to.stat().st_size == expected_size:
                return
        else:
            st.info(f'{url} is already downloaded.')
            if not st.button('Download again?'):
                return
    download_to.parent.mkdir(parents=True, exist_ok=True)
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning('Downloading %s...' % url)
        progress_bar = st.progress(0)

        with open(download_to, 'wb')as output_file:
            with urllib.request.urlopen(url)as response:
                length = int(response.info()['Content-Length'])
                counter = 0.0
                MEGABYTES = 2.0**20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)
                    weights_warning.warning(
                        "Downloading %s... (%6.2f/%6.2f MB)"
                        % (url, counter / MEGABYTES, length / MEGABYTES)
                    )
                    progress_bar.progress(min(counter/length, 1.0))
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()


RTC_CONFIGURATION = RTCConfiguration(
    {'iceServers': [{'urls': ['stun:stun.l.google.com:19302']}]})


def main():
    st.header('Web Arayüzü Demo')

    pages = {
        "Real time object detection": app_object_detection,
        "Real time video transform with simple OpenCV filters ": app_video_filters,
        "Real time audio filter": app_audio_filter,
        "Delayed echo ": app_delayed_echo,
        "Consuming media files on server-side and streaming it to browser": app_streaming,
        "WebRTC is sendonly and images are shown via st.image() ": app_sendonly_video,
        "WebRTC is sendonly and audio frames are visualized with matplotlib ": app_sendonly_audio,
        "Simple video and audio loopback ": app_loopback,
        "Configure media constraints and HTML element styles with loopback": app_media_constraints,
        "Control the playing state programatically": app_programatically_play,
        "Customize UI texts": app_customize_ui_texts,
    }
    page_titles = pages.keys()
    page_title = st.sidebar.selectbox('Choose the app mode', page_titles)
    st.subheader(page_title)
    page_func = pages[page_title]
    page_func()

    logger.debug('=== Alive threads ===')
    for thread in threading.enumerate():
        if thread.is_alive():
            logger.debug(f'{thread.name} ({thread.ident})')


def app_loopback():
    webrtc_streamer(key='loopback')


def app_video_filters():
    _type = st.radio('Select transform type',
                     ('noop', 'cartoon', 'edges', 'rotate'))

    def callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format='bgr24')
        if _type == 'noop':
            pass
        elif _type == 'cartoon':
            img_color = cv2.pvrDown(cv2.pvrDown(img))
            for _ in range(6):
                img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
            img_color = cv2.pvrUp(cv2.pvrUp(img_color))

            img_edges = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_edges = cv2.adaptiveThreshold(cv2.medianBlur(
                img_edges, 7), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
            img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)
            img = cv2.bitwise_and(img_color, img_edges)
        elif _type == 'edges':
            img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)
        elif _type == 'rotate':
            rows, cols, _ = img.shape
            M = cv2.getRotationMatrix2D((cols/2, rows/2), frame.time*45, 1)
            img = cv2.warpAffine(img, M, (cols, rows))
        return av.VideoFrame.from_ndarray(img, format='bgr24')
    webrtc_streamer(
        key='opencv-filter',
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=callback,
        media_stream_constraints={'video': True, 'audio': False},
        async_processing=True
    )


def app_audio_filter():
    gain = st.slider('Gain', -10.0, +20.0, 1.0, 0.05)

    def process_audio(frame: av.AudioFrame) -> av.AudioFrame:
        raw_samples = frame.to_ndarray()
        sound = pydub.AudioSegment(
            data=raw_samples.tobytes(),
            sample_width=frame.format.bytes,
            frame_rate=frame.sample_rate,
            channels=len(frame.layout.channels)
        )
        sound = sound.apply_gain(gain)

        channel_sounds = sound.split_to_mono()
        channel_samples = [s.get_array_of_samples() for s in channel_sounds]
        new_samples: np.ndarray = np.array(channel_samples).T
        new_samples = new_samples.reshape(raw_samples.shape)

        new_frame = av.AudioFrame.from_ndarray(
            new_samples, layout=frame.layout.name)
        new_frame.sample_rate = frame.sample_rate
        return new_frame

    webrtc_streamer(
        key='audio_filter',
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        audio_frame_callback=process_audio,
        async_processing=True
    )


def app_delayed_echo():
    delay = st.slider('Delay', 0.0, 5.0, 1.0, 0.05)

    async def queued_video_frames_callback(frames: List[av.VideoFrame]) -> List[av.VideoFrame]:
        logger.debug('Delay: %f', delay)
        _ = await asyncio.sleep(delay)
        return frames

    async def queued_audio_frames_callback(frames: List[av.AudioFrame]) -> List[av.AudioFrame]:
        _ = await asyncio.sleep(delay)
        return frames

    webrtc_streamer(
        key='delay',
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        queued_video_frames_callback=queued_video_frames_callback,
        queued_audio_frames_callback=queued_audio_frames_callback,
        async_processing=True,
    )


def app_object_detection():
    MODEL_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.caffemodel"
    MODEL_LOCAL_PATH = HERE / './models/MobileNetSSD_deploy.caffemodel'
    PROTOTXT_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.prototxt.txt"
    PROTOTXT_LOCAL_PATH = HERE / './models/MobileNetSSD_deploy.prototxt.txt'
    CLASSES = [
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor", ]

    @st.experimental_singleton
    def generate_label_colors():
        return np.random.uniform(0.255, size=(len(CLASSES), 3))
    COLORS = generate_label_colors()
    download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=23147564)
    download_file(PROTOTXT_URL, PROTOTXT_LOCAL_PATH, expected_size=29353)
    DEFAULT_CONFIDENCE_THRESHOLD = 0.5

    class Detection(NamedTuple):
        name: str
        prob: float

    cache_key = 'object_detection_dnn'
    if cache_key in st.session_state:
        net = st.session_state[cache_key]
    else:
        net = cv2.dnn.readNetFromCaffe(
            str(PROTOTXT_LOCAL_PATH), str(MODEL_LOCAL_PATH))
        st.session_state[cache_key] = net
    confidence_threshold = st.slider(
        'Confidence threshold', 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05)

    def _annotate_image(image, detections):
        h, w = image.shape[:2]
        result: List[Detection] = []
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_threshold:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7]*np.array([w, h, w, h])
                startX, startY, endX, endY = box.astype('int')
                name = CLASSES[idx]
                result.append(Detection(name=name, prob=float(confidence)))

                label = f'{name}: {round(confidence*100,2)}%'
                cv2.rectangle(image, (startX, startY),
                              (endX, endY), COLORS[idx], 2)
                y = startY-15 if startY - 15 > 15 else startY+15
                cv2.putText(image, label, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
        return image, result
    result_queue = (queue.Queue())

    def callback(frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format='bgr24')
        blob = cv2.dnn.blobFromImage(cv2.resize(
            image, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()
        annotated_image, result = _annotate_image(image, detections)
        result_queue.put(result)
        return av.VideoFrame.from_ndarray(annotated_image, format='bgr24')
    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    if st.checkbox('Show the detected labels', value=True):
        if webrtc_ctx.state.playing:
            labels_placeholder = st.empty()
            while True:
                try:
                    result = result_queue.get(timeout=1.0)
                except queue.Empty:
                    result = None
                labels_placeholder.table(result)


def app_streaming():
    MEDIAFILES = {
        "big_buck_bunny_720p_2mb.mp4 (local)": {
            "url": "https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_2mb.mp4",
            "local_file_path": HERE / "data/big_buck_bunny_720p_2mb.mp4",
            "type": "video",
        },
        "big_buck_bunny_720p_10mb.mp4 (local)": {
            "url": "https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_10mb.mp4",
            "local_file_path": HERE / "data/big_buck_bunny_720p_10mb.mp4",
            "type": "video",
        },
        "file_example_MP3_700KB.mp3 (local)": {
            "url": "https://file-examples-com.github.io/uploads/2017/11/file_example_MP3_700KB.mp3",
            "local_file_path": HERE / "data/file_example_MP3_700KB.mp3",
            "type": "audio",
        },
        "file_example_MP3_5MG.mp3 (local)": {
            "url": "https://file-examples-com.github.io/uploads/2017/11/file_example_MP3_5MG.mp3",
            "local_file_path": HERE / "data/file_example_MP3_5MG.mp3",
            "type": "audio",
        },
        "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov": {
            "url": "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov",
            "type": "video",
        },
    }

    media_file_label = st.radio(
        'Select a media source to stream', tuple(MEDIAFILES.keys()))
    media_file_info = MEDIAFILES[media_file_label]
    if 'local_file_path' in media_file_info:
        download_file(media_file_info['url'],
                      media_file_info['local_file_path'])

    def create_player():
        if 'local_file_path' in media_file_info:
            return MediaPlayer(str(media_file_info['local_file_path']))
        else:
            return MediaPlayer(media_file_info['url'])

    key = f'media-streaming-{media_file_label}'
    ctx: Optional[WebRtcStreamerContext] = st.session_state.get(key)
    if media_file_info['type'] == 'video' and ctx and ctx.state.playing:
        _type = st.radio('Select transform type',
                         ('noop', 'cartoon', 'edges', 'rotate'))
    else:
        _type = 'noop'

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        if _type == "noop":
            pass
        elif _type == "cartoon":
            img_color = cv2.pyrDown(cv2.pyrDown(img))
            for _ in range(6):
                img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
            img_color = cv2.pyrUp(cv2.pyrUp(img_color))

            img_edges = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_edges = cv2.adaptiveThreshold(
                cv2.medianBlur(img_edges, 7),
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                9,
                2,
            )
            img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)

            img = cv2.bitwise_and(img_color, img_edges)
        elif _type == "edges":
            img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)
        elif _type == "rotate":
            rows, cols, _ = img.shape
            M = cv2.getRotationMatrix2D(
                (cols / 2, rows / 2), frame.time * 45, 1)
            img = cv2.warpAffine(img, M, (cols, rows))

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key=key,
        mode=WebRtcMode.RECVONLY,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": media_file_info["type"] == "video",
            "audio": media_file_info["type"] == "audio",
        },
        player_factory=create_player,
        video_frame_callback=video_frame_callback,
    )


def app_sendonly_video():
    webrtc_ctx = webrtc_streamer(
        key='video-sendonly',
        mode=WebRtcMode.SENDONLY,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={'video': True}
    )

    image_place = st.empty()
    while True:
        if webrtc_ctx.video_receiver:
            try:
                video_frame = webrtc_ctx.video_receiver.get_frame(timeout=1)
            except queue.Empty:
                logger.warning('Queue is empty. Abort.')
                break
            img_rgb = video_frame.to_ndarray(format='rgb24')
            image_place.image(img_rgb)
        else:
            logger.warning('VideoReceiver is not set. Abort.')
            break


def app_sendonly_audio():
    webrtc_ctx = webrtc_streamer(
        key="sendonly-audio",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=256,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"audio": True},
    )
    fig_place = st.empty()

    fig, [ax_time, ax_freq] = plt.subplots(
        2, 1, gridspec_kw={"top": 1.5, "bottom": 0.2}
    )

    sound_window_len = 5000
    sound_window_buffer = None
    while True:
        if webrtc_ctx.audio_receiver:
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                logger.warning("Queue is empty. Abort.")
                break

            sound_chunk = pydub.AudioSegment.empty()
            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                sound_chunk += sound

            if len(sound_chunk) > 0:
                if sound_window_buffer is None:
                    sound_window_buffer = pydub.AudioSegment.silent(
                        duration=sound_window_len
                    )

                sound_window_buffer += sound_chunk
                if len(sound_window_buffer) > sound_window_len:
                    sound_window_buffer = sound_window_buffer[-sound_window_len:]

            if sound_window_buffer:
                sound_window_buffer = sound_window_buffer.set_channels(1)
                sample = np.array(sound_window_buffer.get_array_of_samples())

                ax_time.cla()
                times = (np.arange(-len(sample), 0)) / \
                    sound_window_buffer.frame_rate
                ax_time.plot(times, sample)
                ax_time.set_xlabel("Time")
                ax_time.set_ylabel("Magnitude")

                spec = np.fft.fft(sample)
                freq = np.fft.fftfreq(
                    sample.shape[0], 1.0 / sound_chunk.frame_rate)
                freq = freq[: int(freq.shape[0] / 2)]
                spec = spec[: int(spec.shape[0] / 2)]
                spec[0] = spec[0] / 2

                ax_freq.cla()
                ax_freq.plot(freq, np.abs(spec))
                ax_freq.set_xlabel("Frequency")
                ax_freq.set_yscale("log")
                ax_freq.set_ylabel("Magnitude")

                fig_place.pyplot(fig)
        else:
            logger.warning("AudioReciver is not set. Abort.")
            break


def app_media_constraints():
    frame_rate = 5
    webrtc_streamer(
        key='media-constraints',
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={
            'video': {'frameRate': {'ideal': frame_rate}}},
        video_html_attrs={'style': {'width': '50%', 'margin': '0 auto', 'border': '5px yellow solid'},
                          'controls': False,
                          'autoPlay': True}
    )
    st.write(f'The frame rate is set as {frame_rate}. Video style is changed.')


def app_programatically_play():
    playing = st.checkbox('Playing', value=True)
    webrtc_streamer(
        key='programatic_control',
        desired_playing_state=playing,
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION
    )


def app_customize_ui_texts():
    webrtc_streamer(
        key="custom_ui_texts",
        rtc_configuration=RTC_CONFIGURATION,
        translations={
            "start": "Start",
            "stop": "Stop",
            "select_device": "Select Device",
            "media_api_not_available": "Media API no Available",
            "device_ask_permission": "Device Ask Permission",
            "device_not_available": "Device not Available",
            "device_access_denied": "Device Access Denied",
        },
    )


if __name__ == '__main__':
    import os
    DEBUG = os.environ.get('DEBUG', 'false').lower() not in [
        'false', 'no', '0']
    logging.basicConfig(format='[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: '
                        '%(message)s',
                        force=True)
    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)
    st_webrtc_logger = logging.getLogger('streamlit_webrtc')
    st_webrtc_logger.setLevel(logging.DEBUG)
    fsevents_logger = logging.getLogger('fsevents')
    fsevents_logger.setLevel(logging.WARNING)
    main()
