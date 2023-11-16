import streamlit as st

st.set_page_config(
    page_title="Welcome To Our Time Series Forecasting App",
    page_icon="😃",
    layout="wide"
)

st.markdown("# 👋 Welcome To Our MLops Project")
st.image("https://www.animatedimages.org/data/media/1645/animated-waving-image-0090.gif")

st.write("""
    <style>
        @keyframes slide-in {
            0% {
                transform: translateX(-100%);
            }
            100% {
                transform: translateX(0);
            }
        }
        .slide-in-animation {
            animation: slide-in 1.5s ease-in-out;
        }
    </style>
""", unsafe_allow_html=True)

st.write('<div class="slide-in-animation">Favorita Corporation is an Ecuador based company engaged in the organization, installation and administration of stores, markets and supermarkets. The company has a business presence in many international countries also. This app with the help of a linear regression model will enable you to predict sales across Favorita Stores.</div>', unsafe_allow_html=True)

st.sidebar.success("Select a page above.")
subheader_container = st.container()
subheader_content = """
<div class="slide-in-animation">
<h3>Things You Can Do On This App:</h3>
<ul>
  <li>Forecast Sales of a Specific Date for Favorita Store</li>
  <li>View the dataset and interact with a visual showing daily sales across stores</li>
  <li>Get to know more about the team behind this app</li>
</ul>
</div>
"""
subheader_container.markdown(subheader_content, unsafe_allow_html=True)

st.write("""
<style>
    @keyframes slide-in {
        0% {
            transform: translateX(-100%);
        }
        100% {
            transform: translateX(0);
        }
    }
    .slide-in-animation {
        animation: slide-in 1.5s ease;
    }
</style>
""", unsafe_allow_html=True)