# Import required libraries
import pandas as pd
import streamlit as st
import os
import glob
from scenedetect import open_video, ContentDetector, SceneManager, StatsManager
from scenedetect.scene_manager import save_images
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import pipeline
import whisper
import moviepy.editor
from PIL import Image


def save_uploaded_file(uploadedfile):
    video_name = uploadedfile.name
    get_video_name = str(video_name).split('.')[0]
    video_directory = "video_frames/" + get_video_name

    if not os.path.exists(video_directory):
        os.makedirs(video_directory)

    with open(os.path.join(video_directory, uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())

    return {
        "directory_path": video_directory,
        "file_path": os.path.join(video_directory, uploadedfile.name)
    }


def predict_with_blip_model(image):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    raw_image = Image.open(image).convert('RGB')

    # unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)


def summarize_with_bart(given_text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer(given_text, do_sample=False)


def check_words_in_transcription(word_list, input_string):
    count = 0
    for word in word_list:
        if word in input_string:
            count += 1

    if count >= 3:
        return True
    return False


def classify_with_bart(given_text):
    classifier = pipeline("zero-shot-classification",
                          model="facebook/bart-large-mnli")

    candidate_labels = ["film_and_animation",
                        "autos_and_vehicles",
                        "music",
                        "pets_and_animals",
                        "sports",
                        "travel_and_events",
                        "gaming",
                        "people_and_blogs",
                        "comedy",
                        "entertainment",
                        "news_and_politics",
                        "how_to_and_style",
                        "education",
                        "science_and_technology",
                        "nonprofits_and_activism"]
    return classifier(given_text, candidate_labels)


def process_video_pipeline(video_directory, video_path):
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

    st.success("Video Frame Processing Started")
    all_video_captions = ""
    for image in all_image_files:
        st.write("Detected Scene : {image_name}".format(image_name=image))
        st.image(image)
        image_caption = predict_with_blip_model(image)
        # detect_objects_in_image(image)
        st.write("In this Image : {captions}".format(captions=image_caption))
        all_video_captions += image_caption + "."
        st.divider()

    st.success("Video Frame Processing Completed")

    st.header("Generated Captions and Video Summary")

    st.subheader("Generated Information From Video")
    st.write("{video_text}".format(video_text=all_video_captions))

    st.subheader("Generated Video Summary from ML Model")
    video_summary = summarize_with_bart(all_video_captions)[0]['summary_text']
    st.write("{summarized_video}".format(summarized_video=video_summary))

    st.header("Video Transcription")

    # Load the Video
    video = moviepy.editor.VideoFileClip(video_path)

    # Extract the Audio
    audio = video.audio

    # Export the Audio
    audio.write_audiofile("audio.mp3")
    model = whisper.load_model("base")
    result = model.transcribe("audio.mp3")
    transcript_text = result["text"]

    st.subheader("Video Transcription from ML Model")
    st.write(transcript_text)

    offensive_words_list = [
        "Arse",
        "Bloody",
        "Bugger",
        "Crap",
        "Minger",
        "Sod - off",
        "Arsehole",
        "Balls",
        "Bint",
        "Bitch",
        "Bollocks",
        "Bullshit",
        "Feck",
        "Munter",
        "Pissed/pissed off",
        "Shit",
        "Son of a bitch",
        "Tits",
        "Bastard",
        "Beef curtains",
        "Bellend",
        "Bloodclaat",
        "Clunge",
        "Cock",
        "Dick",
        "Dickhead",
        "Fanny",
        "Flaps",
        "Gash",
        "Knob",
        "Minge",
        "Prick",
        "Punani",
        "Pussy",
        "Snatch",
        "Twat"
    ]

    flag = check_words_in_transcription(offensive_words_list, transcript_text)
    if not flag:
        st.success("No Offensive Data Found in Video")
        video_classification = classify_with_bart(transcript_text)
        video_results_dataframe = {
            "labels": video_classification['labels'],
            "scores": video_classification['scores']
        }
        video_results_dataframe = pd.DataFrame.from_dict(video_results_dataframe)
        st.table(video_results_dataframe)
    else:
        st.error("Offensive Data Found in Video, Rejecting Video")


# Set the app title
st.title("Video Perception Analyzer App")

# File uploader widget
uploaded_video_file = st.file_uploader("Choose a video mp4 file", type=["mp4"])

# Check if a file has been uploaded
if uploaded_video_file is not None:
    # Display the uploaded video
    st.video(uploaded_video_file)
    saved_video_path = save_uploaded_file(uploaded_video_file)
    st.subheader("Processing Video")
    process_video_pipeline(saved_video_path.get("directory_path"),
                           saved_video_path.get("file_path"))
else:
    st.write("Please upload a video file")
