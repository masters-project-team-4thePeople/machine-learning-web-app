# Import required libraries
import streamlit as st
import os
import glob
from scenedetect import open_video, ContentDetector, SceneManager, StatsManager
from scenedetect.scene_manager import save_images


def find_scenes(video_path):
    st.write("Processing File {video_path}".format(video_path=video_path))
    video_name = video_path.name
    get_video_name = str(video_name).split('.')[0]
    video_directory = "video_frames/" + get_video_name

    if not os.path.exists(video_directory):
        os.makedirs(video_directory)

    video_stream = open_video(video_path)
    stats_manager = StatsManager()

    # Construct our SceneManager and pass it our StatsManager.
    scene_manager = SceneManager(stats_manager)
    scene_manager.add_detector(ContentDetector(threshold=65))

    # Perform scene detection.
    scene_manager.detect_scenes(video=video_stream)
    scene_list = scene_manager.get_scene_list()

    # Store the frame metrics we calculated for the next time the program runs.
    save_images(scene_list, video_stream, num_images=1, show_progress=True, output_dir=video_directory)

    all_image_files = glob.glob(video_directory + '/*.jpg')
    all_image_files.sort()

    video_understanding = ""
    for image in all_image_files:
        st.write(" Processing {image_name}".format(image_name=image))


# Set the app title
st.title("Video Perception Analyzer App")

# File uploader widget
uploaded_video_file = st.file_uploader("Choose a video mp4 file", type=["mp4"])

# Check if a file has been uploaded
if uploaded_video_file is not None:
    # Display the uploaded video
    st.video(uploaded_video_file)
    find_scenes(uploaded_video_file)
else:
    st.write("Please upload a video file")


