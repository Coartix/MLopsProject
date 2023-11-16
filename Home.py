import streamlit as st

st.set_page_config(
    page_title="Welcome To Our Semantic Segmentation App",
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

st.write('<div class="slide-in-animation">This is an app used for semantic segmentation trained on Pascal VOC 2007 made up to 20 classes.<br/>Each image can contain multiple objects.</div>', unsafe_allow_html=True)
st.write('<div class="slide-in-animation">Here are the 20 classes:</div>', unsafe_allow_html=True)
st.write('<div class="slide-in-animation">1. Person</div>', unsafe_allow_html=True)
st.write('<div class="slide-in-animation">2. Bird</div>', unsafe_allow_html=True)
st.write('<div class="slide-in-animation">3. Cat</div>', unsafe_allow_html=True)
st.write('<div class="slide-in-animation">4. Cow</div>', unsafe_allow_html=True)
st.write('<div class="slide-in-animation">5. Dog</div>', unsafe_allow_html=True)
st.write('<div class="slide-in-animation">6. Horse</div>', unsafe_allow_html=True)
st.write('<div class="slide-in-animation">7. Sheep</div>', unsafe_allow_html=True)
st.write('<div class="slide-in-animation">8. Aeroplane</div>', unsafe_allow_html=True)
st.write('<div class="slide-in-animation">9. Bicycle</div>', unsafe_allow_html=True)
st.write('<div class="slide-in-animation">10. Boat</div>', unsafe_allow_html=True)
st.write('<div class="slide-in-animation">11. Bus</div>', unsafe_allow_html=True)
st.write('<div class="slide-in-animation">12. Car</div>', unsafe_allow_html=True)
st.write('<div class="slide-in-animation">13. Motorbike</div>', unsafe_allow_html=True)
st.write('<div class="slide-in-animation">14. Train</div>', unsafe_allow_html=True)
st.write('<div class="slide-in-animation">15. Bottle</div>', unsafe_allow_html=True)
st.write('<div class="slide-in-animation">16. Chair</div>', unsafe_allow_html=True)
st.write('<div class="slide-in-animation">17. Dining Table</div>', unsafe_allow_html=True)
st.write('<div class="slide-in-animation">18. Potted Plant</div>', unsafe_allow_html=True)
st.write('<div class="slide-in-animation">19. Sofa</div>', unsafe_allow_html=True)
st.write('<div class="slide-in-animation">20. TV/Monitor</div>', unsafe_allow_html=True)

subheader_container = st.container()
subheader_content = """
<div class="slide-in-animation">
<h3>Things You Can Do On This App:</h3>
<ul>
  <li>Use our semantic segmentation on images containing such classes</li>
  <li>View the dataset and train the model</li>
  <li>Get to know more about the team behind this app</li>
</ul>
</div>
"""
subheader_container.markdown(subheader_content, unsafe_allow_html=True)


st.sidebar.success("Select a page above.")
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