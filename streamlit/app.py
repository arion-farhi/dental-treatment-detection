import streamlit as st
import json
import pandas as pd
from PIL import Image
import plotly.express as px

st.set_page_config(page_title="Dental Treatment Detection", layout="wide")

# Load metrics
with open('model_metrics.json', 'r') as f:
    metrics = json.load(f)

st.title("Dental Treatment Detection with AutoML")
st.markdown("**Object Detection Model** - Automated identification of dental treatments from panoramic X-rays")

# Sidebar
st.sidebar.header("Model Info")
st.sidebar.metric("Model Type", "AutoML Vision")
st.sidebar.metric("Overall mAP", f"{metrics['overall']['mAP']:.1%}")
st.sidebar.metric("Training Budget", "20 node hours")

st.sidebar.markdown("---")
st.sidebar.markdown("### Dataset")
st.sidebar.markdown(f"- Total Images: {metrics['overall']['total_images']:,}")
st.sidebar.markdown(f"- Training: {metrics['overall']['training_images']:,}")
st.sidebar.markdown(f"- Validation: {metrics['overall']['validation_images']}")
st.sidebar.markdown(f"- Test: {metrics['overall']['test_images']}")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Sample Detections", "Model Performance", "Training Details", "Key Findings"])

with tab1:
    st.header("Sample Detections")
    st.markdown("Model detections on dental X-ray images showing identified treatments:")
    
    img = Image.open('images/sample_detections.png')
    st.image(img, use_container_width=True)
    
    st.markdown("### Detected Classes")
    col1, col2, col3, col4, col5 = st.columns(5)
    classes = ["Cavity", "Fillings", "Impacted Tooth", "Implant", "Infected-teeth"]
    for col, cls in zip([col1, col2, col3, col4, col5], classes):
        col.metric(cls, "✓")

with tab2:
    st.header("Model Performance")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Avg Precision", f"{metrics['overall']['mAP']:.1%}")
    col2.metric("Precision", f"{metrics['overall']['precision']:.1%}")
    col3.metric("Recall", f"{metrics['overall']['recall']:.1%}")
    
    st.markdown("---")
    
    # Per-class performance chart
    st.subheader("Per-Class Performance")
    img_perf = Image.open('images/per_class_performance.png')
    st.image(img_perf, use_container_width=True)
    
    st.markdown("---")
    
    # Class size vs performance
    st.subheader("Impact of Training Data Size")
    img_size = Image.open('images/class_size_vs_performance.png')
    st.image(img_size, use_container_width=True)
    
    # Performance table
    st.markdown("---")
    st.subheader("Detailed Metrics")
    df = pd.DataFrame(metrics['per_class'])
    df.columns = ['Class', 'Average Precision', 'Training Samples']
    df['AP %'] = (df['Average Precision'] * 100).round(1)
    df = df[['Class', 'AP %', 'Training Samples']]
    st.dataframe(df, hide_index=True, use_container_width=True)

with tab3:
    st.header("Training Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Model Configuration")
        st.markdown("""
        - **Model Type:** AutoML CLOUD_HIGH_ACCURACY_1
        - **Training Budget:** 20 node hours (~$63)
        - **Training Time:** 4-6 hours (wall time)
        - **Platform:** Google Cloud Vertex AI
        - **Region:** us-central1
        """)
        
        st.markdown("### Dataset")
        st.markdown("""
        - **Source:** DentAi (Roboflow)
        - **Total Images:** 9,772 (after augmentation)
        - **Classes:** 5 (Cavity, Fillings, Impacted Tooth, Implant, Infected-teeth)
        - **Annotations:** 61,720 bounding boxes
        - **Split:** 94% train, 4% validation, 2% test
        """)
    
    with col2:
        st.markdown("### Architecture Pipeline")
        st.code("""
1. Data Preparation
   ↓
2. Upload to GCS
   ↓
3. Create Vertex Dataset
   ↓
4. AutoML Training
   ↓
5. Model Evaluation
   ↓
6. Model Registry
        """, language=None)
        
        st.markdown("### Tech Stack")
        st.markdown("""
        - **ML Platform:** Vertex AI AutoML
        - **Storage:** Google Cloud Storage
        - **Data Processing:** Pandas, Roboflow API
        - **Visualization:** Matplotlib, Pillow
        - **Notebooks:** Jupyter, Vertex AI Workbench
        """)

with tab4:
    st.header("Key Findings & Insights")
    
    st.markdown("### 1. Class Imbalance Impact")
    st.markdown("""
    The model's performance closely correlates with training data size:
    - **Best performers:** Implant (75.1% AP) and Infected-teeth (74.6% AP) had 10K+ samples
    - **Worst performer:** Cavity (42.9% AP) with only 3,456 samples (6% of data)
    - **Recommendation:** Collect more cavity examples or apply targeted augmentation
    """)
    
    st.markdown("### 2. Fillings Underperformance")
    st.markdown("""
    Despite being 54% of the dataset (31,434 samples), Fillings only achieved 69.5% AP:
    - **Root cause:** High intra-class variation (new vs. old fillings look very different)
    - **Solution:** May need class splitting (new_fillings vs. old_fillings)
    """)
    
    st.markdown("### 3. Training Budget Sweet Spot")
    st.markdown("""
    20 node hours achieved 70.9% mAP at ~$63 cost:
    - 40 hours would add only 2-4 mAP points for $60 more (diminishing returns)
    - AutoML used early stopping, actual cost likely $50-60
    - **Cost-effective choice for portfolio project**
    """)
    
    st.markdown("### 4. Production Readiness")
    st.markdown("""
    - **Ready:** Implant and Infected-teeth detection (>74% AP)
    - **Needs improvement:** Cavity detection (more data needed)
    - **Consider ensemble:** Combine with rule-based fallbacks for low-confidence predictions
    """)
    
    st.markdown("---")
    st.markdown("### Next Steps")
    st.markdown("""
    1. Collect more cavity and impacted tooth training examples
    2. Investigate fillings class split (new vs. old)
    3. Test on real clinical data (with HIPAA compliance)
    4. Implement ensemble with dentist-in-the-loop review
    """)

st.markdown("---")
st.markdown("**Project by Arion Farhi** | [GitHub](https://github.com/arion-farhi)")
