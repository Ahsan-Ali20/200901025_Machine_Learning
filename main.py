import streamlit as st
import pickle
import numpy as np

# Load the pre-trained model and data
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('data.pkl', 'rb'))

# Set page title and icon
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide",  # Enable wide mode for better spacing
)

# Apply the CSS for title and sidebar
title_css = """
h1 {
    text-align: center;
    font-size: 5.0em !important;
    color: black;
    font-weight: bold;
    
}

.subheader {
    font-size: 5em !important;
    color: black;
}
"""

# Read and apply the CSS
css = """
.stApp > header {
    background-color: transparent;
}

.stApp {
  margin: auto;
  font-family: -apple-system, BlinkMacSystemFont, sans-serif;
  overflow: auto;
  background: linear-gradient(315deg, #022b70 68%, #ebf0ff 98%);
  animation: gradient 15s ease infinite;
  background-size: 400% 300%;
  background-attachment: fixed;
  
}

.stApp .sidebar-container {
  background: linear-gradient(45deg, #2e6ad3 3%, #293991 38%, #7dc4ff 68%, #cfa436 98%);
  border-radius: 10px;
  box-shadow: 0 4px 8px rgba(255, 255, 255, 0.1);
  padding: 20px;
  margin: 20px 0;
}

.stButton {
  background: linear-gradient(100deg, #130046, #7698ff);
  color: rgb(0, 38, 255);
  border-radius: 25px;
  padding: 20px 30px; /* Adjust the padding to increase the button size */
  cursor: pointer;
}

.stButton:hover {
  background: linear-gradient(90deg, #710024, #ff51c2);
  color: #ff51c2; /* Change text color on hover, if needed */
}


.stPanel {
  background: linear-gradient(45deg, #0e0ae3, #910000);
  border-radius: 25px;
  box-shadow: 0 2px 4px rgba(249, 255, 69, 0.1);
  padding: 20px;
  margin: 20px 0;
}


"""

# Apply the CSS
st.markdown(f'<style>{css + title_css}</style>', unsafe_allow_html=True)
st.write("----------------------------------------------------------------------------------------")
# Page title and description
st.title("Laptop Price Predictor")
st.write("----------------------------------------------------------------------------------------")

st.write("Fill the input details to predict the price of a laptop.")

# Sidebar with user inputs
with st.sidebar:
    st.markdown("<h2 style='text-align: center; font-size: 35px; font-weight: bold;'><b>User Inputs</b></h2>", unsafe_allow_html=True)
    st.write("-----------------------")
    # Brand
    company = st.selectbox('Brand', df['Company'].unique())
    # Type of laptop
    laptop_type = st.selectbox('Type', df['TypeName'].unique())
    # Ram
    ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
    # Weight
    weight = st.number_input("Weight of the laptop", min_value=0.0)
    # Touchscreen
    touchscreen = st.selectbox('TouchScreen', ['No', 'Yes'])
    # IPS
    ips = st.selectbox('IPS', ['No', 'Yes'])
    # Screen size
    screen_size = st.number_input('Screen Size', min_value=1.0)
    # Resolution
    resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160',
                                                    '3200x1800', '2880x1800', '2560x1600', '2560x1440',
                                                    '2304x1440'])
    # CPU
    cpu = st.selectbox('CPU', df['Cpu brand'].unique())
    # HDD
    hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
    # SSD
    ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])
    # GPU
    gpu = st.selectbox('GPU', df['Gpu brand'].unique())
    # OS
    os = st.selectbox('Operating System', df['os'].unique())

st.markdown(
    """
    <style>
        .stButton {
            font-size: 30px;
            padding: 15px 25px; /* Adjust padding to increase overall size */
        }
    </style>
    """,
    unsafe_allow_html=True
)


# Predict button
if st.button('## **Click Here, To Predict Price**'):
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    x_res = int(resolution.split('x')[0])
    y_res = int(resolution.split('x')[1])
    ppi = ((x_res ** 2) + (y_res ** 2)) ** 0.5 / screen_size
    query = np.array([company, laptop_type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os],
                     dtype=object)
    query = query.reshape(1, 12)

    # Predicted Price
    predicted_price = int(np.exp(pipe.predict(query)))

    # Create a container for the predicted price with gradient styling using st.markdown
    st.markdown(
        f"<div style='background: linear-gradient(45deg, rgb(10 0 100) 3%, rgb(72 90 201) 38%, rgb(136 119 255) 68%, rgb(21 21 21) 98%); "
        f"border-radius: 10px; box-shadow: 0 4px 8px rgba(255, 255, 255, 0.1); padding: 20px; margin: 20px 0; "
        f"color: white; text-align: center;'>"
        f"<div class='stPanel' style='font-size: 30px; font-weight: bold;'>"
        f"Predicted Price is RS. {predicted_price}"
        f"</div>",
        unsafe_allow_html=True,
    )
# Display images
st.markdown("""
    <div style="text-align: center;">
        <h2>Featured Laptops</h2>
    </div>
""", unsafe_allow_html=True)
st.write("### **----------------------------------------------------------------------------------------**")
# Create a list of image paths (replace with your actual paths)
image_file_paths = [
    "/content/drive/MyDrive/200901025_ML_PROJECT/Ahsan_Ali_Machine_Learning_Project/images/a1.png",
    "/content/drive/MyDrive/200901025_ML_PROJECT/Ahsan_Ali_Machine_Learning_Project/images/b1.png",
    "/content/drive/MyDrive/200901025_ML_PROJECT/Ahsan_Ali_Machine_Learning_Project/images/c1.png",
    "/content/drive/MyDrive/200901025_ML_PROJECT/Ahsan_Ali_Machine_Learning_Project/images/d1.png"
]

# Use containers for layout
col1, col2 = st.columns(2)

# Set a custom width for the images
image_width = 300
image_height = 200

for i, image_path in enumerate(image_file_paths):
    with col1 if i < 2 else col2:
        st.image(image_path, caption=f"Laptop {i+1}", width=image_width)
        st.write("----------------------------------------------------------------------------------------")
st.write("=================================================================================================")
st.markdown("""
    <div style="text-align: center;">
        <h4>Desingned By: Muhammad Ahsan Ali</h1>
        <h6>Course: Machine Learning</h6>
        <h6>BSCSO1 Section-A</h6>
        <h6>Registration No: 200901025</h6>
    </div>
""", unsafe_allow_html=True)
st.write("=================================================================================================")