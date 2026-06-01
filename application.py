
import streamlit as st
import joblib
from datetime import datetime

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="AI Support Ticket Classifier",
    page_icon="🎫",
    layout="wide"
)

# ==========================
# LOAD MODEL
# ==========================

model = joblib.load("ticket_classifier.pkl")

# ==========================
# SIDEBAR
# ==========================

st.sidebar.title("📌 Project Information")

st.sidebar.info("""
**Project Name**
AI-Powered Support Ticket Classification

**Model**
LinearSVC + TF-IDF

**Accuracy**
84.3%

**Dataset**
47,837 Support Tickets
""")

st.sidebar.markdown("---")

st.sidebar.success(
    f"Date: {datetime.now().strftime('%d-%m-%Y')}"
)

# ==========================
# MAIN HEADER
# ==========================

st.title("🎫 AI Support Ticket Classification System")

st.markdown("""
Automatically classify customer support tickets into categories such as:

✅ Access  
✅ Hardware  
✅ HR Support  
✅ Storage  
✅ Purchase  
✅ Administrative Rights  
✅ Internal Project  
✅ Miscellaneous
""")

st.markdown("---")

# ==========================
# INPUT SECTION
# ==========================

col1, col2 = st.columns([2, 1])

with col1:

    ticket_text = st.text_area(
        "📝 Enter Support Ticket",
        height=200,
        placeholder="""
Example:
Unable to connect to company VPN after password reset.
        """
    )

with col2:

    st.subheader("📚 Sample Tickets")

    st.info(
        "Unable to access VPN after password reset"
    )

    st.info(
        "Need approval for software purchase"
    )

    st.info(
        "Laptop hard disk not detected"
    )

# ==========================
# PREDICTION
# ==========================

if st.button("🚀 Predict Category"):

    if ticket_text.strip() == "":

        st.warning(
            "Please enter a ticket description."
        )

    else:

        prediction = model.predict([ticket_text])[0]

        st.success(
            f"🎯 Predicted Category: **{prediction}**"
        )


# ==========================
# PROJECT METRICS
# ==========================

st.markdown("---")

st.subheader("📊 Model Performance")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "Accuracy",
        "84.3%"
    )

with c2:
    st.metric(
        "Dataset Size",
        "47,837"
    )

with c3:
    st.metric(
        "Categories",
        "8"
    )

# ==========================
# FOOTER
# ==========================

st.markdown("---")

st.markdown(
"""
### 👨‍💻 Developed By

**Shakith Shetty**

"""
)
