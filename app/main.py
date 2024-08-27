import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import scipy.io
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
from subplotted import subplotted

def plot_ecg(uploaded_ecg, FS):
    '''
    Visualize the ECG signal. 
    
    Parameters
    ----------
    uploaded_ecg : numpy.ndarray
        The ECG signal as a numpy array.
    FS : int
        The sampling frequency of the ECG signal.
    
    Returns
    -------
        The figure object created by matplotlib of the ECG signal. 
    '''
    ecg_1d = uploaded_ecg.reshape(-1)
    N = len(ecg_1d)
    time = np.arange(N) / FS
    p = FS * 5

    for S, ax, i in subplotted(int(np.ceil(len(time) / p)), ncols=2, figsize=(8, 10)):
        segment = ecg_1d[i * p:(i * p + p)]
        time_segment = time[i * p:(i * p + p)]
        ax.plot(time_segment, segment)
        ax.set_title(f'Segment from {i * 5} to {5 * i + 5} seconds', fontsize=7)
        ax.set_xlabel('Time in s', fontsize=5)
        ax.set_ylabel('ECG in mV', fontsize=5)
        ax.set_ylim([segment.min() - .5, segment.max() + .5])
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(1))

        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))

        ax.grid(which='major', color='#CCCCCC', linestyle='--')
        ax.grid(which='minor', color='#CCCCCC', linestyle=':')
        ax.xaxis.set_tick_params(labelsize=6)
        ax.yaxis.set_tick_params(labelsize=6)

    S.fig.tight_layout()
    return S.fig

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(
    page_title='🫀 ECG Classification',
    # anatomical heart favicon
    page_icon="https://api.iconify.design/openmoji/anatomical-heart.svg?width=500",
    layout='wide'
)

# Add a header picture
image_path = "ncai header.png"  # Provide the path to your image here
st.image(image_path, use_column_width=True)
# Add a header above the title


# Page Intro
st.write("""
# 🫀 ECG Classification



**Possible Predictions:** Atrial Fibrillation, Normal, Other Rhythm, or Noise


**Try uploading your own ECG!**

-------
""".strip())

#---------------------------------#
# Data preprocessing and Model building

@st.cache_data
def read_ecg_preprocessing(uploaded_ecg):
    FS = 300
    maxlen = 30 * FS

    uploaded_ecg.seek(0)
    mat = scipy.io.loadmat(uploaded_ecg)
    mat = mat["val"][0]

    uploaded_ecg = np.array([mat])

    X = np.zeros((1, maxlen))
    uploaded_ecg = np.nan_to_num(uploaded_ecg)  # removing NaNs and Infs
    uploaded_ecg = uploaded_ecg[0, 0:maxlen]
    uploaded_ecg = uploaded_ecg - np.mean(uploaded_ecg)
    uploaded_ecg = uploaded_ecg / np.std(uploaded_ecg)
    X[0, :len(uploaded_ecg)] = uploaded_ecg.T  # padding sequence
    uploaded_ecg = X
    uploaded_ecg = np.expand_dims(uploaded_ecg, axis=2)
    return uploaded_ecg

model_path = 'models/weights-best.hdf5'
classes = ['Normal', 'Atrial Fibrillation', 'Other', 'Noise']

@st.cache_resource
def get_model(model_path):
    model = load_model(f'{model_path}')
    return model

@st.cache_resource
def get_prediction(data, _model):
    prob = _model(data)
    ann = np.argmax(prob)
    return classes[ann], prob

# Visualization --------------------------------------
@st.cache_resource
def visualize_ecg(ecg, FS):
    fig = plot_ecg(uploaded_ecg=ecg, FS=FS)
    return fig

# Formatting ---------------------------------#

hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {    
            visibility: hidden;
        }
        footer:after {
            content:'Made for Machine Learning in Healthcare with Streamlit';
            visibility: visible;
            display: block;
            position: relative;
            padding: 5px;
            top: 2px;
        }
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

#---------------------------------#
# Sidebar - Collects user input features into dataframe
with st.sidebar.header('1. Upload your ECG'):
    uploaded_file = st.sidebar.file_uploader("Upload your ECG in .mat format", type=["mat"])

st.sidebar.markdown("")

file_gts = {
    "A00001": "Normal",
    "A00002": "Normal",
    "A00003": "Normal",
    "A00004": "Atrial Fibrilation",
    "A00005": "Other",
    "A00006": "Normal",
    "A00007": "Normal",
    "A00008": "Other",
    "A00009": "Atrial Fibrilation",
    "A00010": "Normal",
    "A00015": "Atrial Fibrilation",
    "A00205": "Noise",
    "A00022": "Noise",
    "A00034": "Noise",
}
valfiles = [
    'None',
    'A00001.mat', 'A00010.mat', 'A00002.mat', 'A00003.mat',
    "A00022.mat", "A00034.mat", 'A00009.mat', "A00015.mat",
    'A00008.mat', 'A00006.mat', 'A00007.mat', 'A00004.mat',
    "A00205.mat", 'A00005.mat'
]

if uploaded_file is None:
    with st.sidebar.header('2. Or use a file from the validation set'):
        pre_trained_ecg = st.sidebar.selectbox(
            'Select a file from the validation set',
            valfiles,
            format_func=lambda x: f'{x} ({(file_gts.get(x.replace(".mat", "")))})' if ".mat" in x else x,
            index=1,
        )
        if pre_trained_ecg != "None":
            f = open("data/validation/" + pre_trained_ecg, 'rb')
            if not uploaded_file:
                uploaded_file = f
        
else:
    st.sidebar.markdown("Remove the file above to demo using the validation set.")

st.sidebar.markdown("---------------")

#---------------------------------#
# Main panel

model = get_model(f'{model_path}')

if uploaded_file is not None:
    col1, _, col2 = st.columns((0.5, .05, 0.45))

    with col1:  # visualize ECG
        st.subheader('1. Visualize ECG')
        ecg = read_ecg_preprocessing(uploaded_file)
        
        fig = visualize_ecg(ecg, FS=300)
        st.pyplot(fig, use_container_width=True)

    with col2:  # classify ECG
        st.subheader('2. Model Predictions')
        with st.spinner(text="Running Model..."):
            pred, conf = get_prediction(ecg, model)
        mkd_pred_table = [
            "| Rhythm Type | Confidence |",
            "| --- | --- |"
        ]
        for i in range(len(classes)):
            mkd_pred_table.append(f"| {classes[i]} | {conf[0, i] * 100:3.1f}% |")
        mkd_pred_table = "\n".join(mkd_pred_table)

        st.write("ECG classified as **{}**".format(pred))
        pred_confidence = conf[0, np.argmax(conf)] * 100
        st.write("Confidence of the prediction: **{:3.1f}%**".format(pred_confidence))
        st.write(f"**Likelihoods:**")
        st.markdown(mkd_pred_table)
