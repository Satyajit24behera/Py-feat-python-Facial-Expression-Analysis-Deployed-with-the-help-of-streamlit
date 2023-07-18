import streamlit as st
from PIL import Image
import os
from feat.utils.io import get_test_data_path
from feat.plotting import imshow
import os
from feat.utils.io import read_feat
import matplotlib.pyplot as plt
from feat import Detector
import base64

st.image('logo.png')
detector = Detector(
    face_model="retinaface",
    landmark_model="mobilefacenet",
    au_model='xgb',
    emotion_model="resmasknet",
    facepose_model="img2pose",
)

detector
source = ["Image", "Video"] 
source_index = st.sidebar.radio("Select the input source:", range(
        len(source)), format_func=lambda x: source[x]) 
is_valid=False

if source_index == 0:
# Create a file uploader in Streamlit
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # Check if an image has been uploaded
    if uploaded_file is not None:
        
        # Open the uploaded image
        image = Image.open(uploaded_file)

        # Display the uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Extract the filename from the uploaded file
        filename = os.path.basename(uploaded_file.name)

        # Specify the directory to save the image
        save_directory = "C:\\Users\\Satyajit\\Downloads\\Documents\\New folder"

        # Save the image to the specified directory with the original filename
        image.save(os.path.join(save_directory, filename))

        single_face_img_path = os.path.join(save_directory, filename)
        imshow(single_face_img_path)
        st.write("image saved at: ",single_face_img_path)
        single_face_prediction = detector.detect_image(single_face_img_path)

        single_face_prediction
        single_face_prediction.to_csv("output3t6.csv", index=False)


        input_prediction = read_feat("output3t6.csv")


        input_prediction
        figs = single_face_prediction.plot_detections(poses=True)
        
    
        for fig in figs:
            # Display the figure using Streamlit
            st.pyplot(fig)
        
        figs = single_face_prediction.plot_detections(faces='aus', muscles=True)
        
        for fig in figs:
            # Display the figure using Streamlit
            st.pyplot(fig)
             
        file_path = "output3t6.csv"
 
        if st.button('Download File'):
            with open(file_path, 'rb') as file:
                file_content = file.read()
            b64 = base64.b64encode(file_content).decode()
            href = f'<a href="data:file/txt;base64,{b64}" download="{file_path.split("/")[-1]}">Click here to download</a>'
            st.markdown(href, unsafe_allow_html=True)
            
elif source_index == 1:
    uploaded_file = st.sidebar.file_uploader("Upload a Video", type=['mp4', 'mpeg', 'mov'])
    
    if uploaded_file is not None:
        # Extract the filename from the uploaded file
        filename = os.path.basename(uploaded_file.name)

        # Specify the directory to save the video
        save_directory = "C:\\Users\\Satyajit\\Downloads\\Documents\\New folder"

        # Save the video to the specified directory with the original filename
        save_path = os.path.join(save_directory, filename)

        # Save the uploaded file to the specified path
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        #to display the video
        st.video(save_path)

        st.success("Video saved successfully!")
        st.write("Video saved at:", save_path)
        
        video_prediction = detector.detect_video(save_path,skip_frames=24)
        
        video_prediction
        input_prediction = read_feat("output3t6.csv")


        input_prediction
        
        video_prediction.shape
        
        figs= video_prediction.loc[[48, 96]].plot_detections(faceboxes=False, add_titles=False)
        for fig in figs:
            # Display the figure using Streamlit
            st.pyplot(fig)
        
        axes = video_prediction.emotions.plot()         
       
        st.pyplot(plt)
        
        file_path = "output3t6.csv"
 
        if st.button('Download File'):
            with open(file_path, 'rb') as file:
                file_content = file.read()
            b64 = base64.b64encode(file_content).decode()
            href = f'<a href="data:file/txt;base64,{b64}" download="{file_path.split("/")[-1]}">Click here to download</a>'
            st.markdown(href, unsafe_allow_html=True)
