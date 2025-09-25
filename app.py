import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import time
import threading
from fact_checker import FactChecker
from pathway_pipeline import PathwayPipeline
from utils import format_confidence_score, get_verification_color, load_sample_sources

# Page configuration
st.set_page_config(
    page_title="Real-Time Fact Checker",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'fact_checker' not in st.session_state:
    st.session_state.fact_checker = FactChecker()
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = PathwayPipeline()
if 'processed_claims' not in st.session_state:
    st.session_state.processed_claims = []
if 'monitoring_data' not in st.session_state:
    st.session_state.monitoring_data = []

def main():
    # Title and description
    st.title("🔍 Real-Time Misinformation Detection & Fact-Checking")
    st.markdown("""
    This platform uses **Pathway** for real-time streaming data processing and **RAG-based verification** 
    to detect misinformation and fact-check claims in real-time.
    """)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select a page:",
        ["Fact Check Claims", "Live Monitoring", "How It Works", "Analytics"]
    )
    
    if page == "Fact Check Claims":
        fact_check_page()
    elif page == "Live Monitoring":
        live_monitoring_page()
    elif page == "How It Works":
        how_it_works_page()
    elif page == "Analytics":
        analytics_page()

def fact_check_page():
    st.header("📝 Submit Claims for Fact-Checking")
    
    # Input methods
    input_method = st.radio(
        "Choose input method:",
        ["Text Input", "File Upload"]
    )
    
    claim_text = ""
    
    if input_method == "Text Input":
        claim_text = st.text_area(
            "Enter a claim or news article to fact-check:",
            placeholder="Example: 'Vaccines contain microchips for tracking people'",
            height=150
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload a text file containing claims:",
            type=['txt', 'md']
        )
        if uploaded_file is not None:
            claim_text = str(uploaded_file.read(), "utf-8")
            st.text_area("Uploaded content:", claim_text, height=150, disabled=True)
    
    # Processing options
    col1, col2 = st.columns(2)
    with col1:
        check_sentiment = st.checkbox("Include sentiment analysis", value=True)
    with col2:
        detailed_analysis = st.checkbox("Detailed source analysis", value=False)
    
    # Fact-check button
    if st.button("🔍 Fact-Check This Claim", type="primary"):
        if claim_text.strip():
            with st.spinner("Processing claim through Pathway pipeline..."):
                # Process through Pathway pipeline
                pipeline_result = st.session_state.pipeline.process_claim(claim_text)
                
                # Perform fact-checking
                fact_check_result = st.session_state.fact_checker.verify_claim(
                    claim_text, 
                    include_sentiment=check_sentiment,
                    detailed_analysis=detailed_analysis
                )
                
                # Store result
                result = {
                    'timestamp': datetime.now(),
                    'claim': claim_text[:100] + "..." if len(claim_text) > 100 else claim_text,
                    'full_claim': claim_text,
                    'verification_status': fact_check_result['status'],
                    'confidence_score': fact_check_result['confidence'],
                    'sources': fact_check_result['sources'],
                    'pipeline_data': pipeline_result
                }
                
                st.session_state.processed_claims.append(result)
                st.session_state.monitoring_data.append({
                    'timestamp': datetime.now(),
                    'status': fact_check_result['status'],
                    'confidence': fact_check_result['confidence']
                })
                
                # Display results
                display_fact_check_results(fact_check_result, pipeline_result)
        else:
            st.warning("Please enter a claim to fact-check.")

def display_fact_check_results(fact_result, pipeline_result):
    st.subheader("✅ Fact-Check Results")
    
    # Main verification status
    status_color = get_verification_color(fact_result['status'])
    confidence_formatted = format_confidence_score(fact_result['confidence'])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Verification Status", 
            fact_result['status'].title(),
            delta=None
        )
    with col2:
        st.metric(
            "Confidence Score", 
            f"{confidence_formatted}%"
        )
    with col3:
        st.metric(
            "Sources Found", 
            len(fact_result['sources'])
        )
    
    # Status indicator
    status_emoji = {"verified": "✅", "false": "❌", "uncertain": "⚠️", "unverified": "❓"}
    st.markdown(f"""
    ### {status_emoji.get(fact_result['status'], '❓')} {fact_result['status'].title()}
    **Explanation:** {fact_result.get('explanation', 'Analysis complete.')}
    """)
    
    # Pipeline processing details
    if pipeline_result:
        with st.expander("🔧 Pathway Pipeline Details"):
            st.json(pipeline_result)
    
    # Sources and citations
    if fact_result['sources']:
        st.subheader("📚 Sources and Citations")
        for i, source in enumerate(fact_result['sources'][:5], 1):
            with st.expander(f"Source {i}: {source.get('title', 'Unknown Title')}"):
                st.write(f"**Relevance Score:** {source.get('relevance', 0):.2f}")
                st.write(f"**Summary:** {source.get('summary', 'No summary available.')}")
                if source.get('url'):
                    st.write(f"**URL:** [Link]({source['url']})")
    
    # Sentiment analysis (if available)
    if fact_result.get('sentiment'):
        st.subheader("💭 Sentiment Analysis")
        sentiment_data = fact_result['sentiment']
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sentiment", sentiment_data.get('label', 'Neutral'))
        with col2:
            st.metric("Sentiment Score", f"{sentiment_data.get('score', 0):.2f}")

def live_monitoring_page():
    st.header("📊 Live Monitoring Feed")
    st.markdown("Real-time view of recently processed claims and their verification status.")
    
    # Auto-refresh toggle
    auto_refresh = st.checkbox("Auto-refresh every 5 seconds", value=False)
    
    if auto_refresh:
        time.sleep(5)
        st.rerun()
    
    # Manual refresh button
    if st.button("🔄 Refresh Data"):
        st.rerun()
    
    # Recent claims table
    if st.session_state.processed_claims:
        st.subheader("Recent Claims")
        
        # Convert to DataFrame for display
        df_data = []
        for claim in st.session_state.processed_claims[-10:]:  # Show last 10
            df_data.append({
                'Time': claim['timestamp'].strftime('%H:%M:%S'),
                'Claim Preview': claim['claim'],
                'Status': claim['verification_status'],
                'Confidence': f"{format_confidence_score(claim['confidence_score'])}%",
                'Sources': len(claim['sources'])
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True)
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        total_claims = len(st.session_state.processed_claims)
        verified_claims = len([c for c in st.session_state.processed_claims if c['verification_status'] == 'verified'])
        false_claims = len([c for c in st.session_state.processed_claims if c['verification_status'] == 'false'])
        uncertain_claims = len([c for c in st.session_state.processed_claims if c['verification_status'] == 'uncertain'])
        
        with col1:
            st.metric("Total Claims", total_claims)
        with col2:
            st.metric("Verified", verified_claims)
        with col3:
            st.metric("False", false_claims)
        with col4:
            st.metric("Uncertain", uncertain_claims)
    else:
        st.info("No claims have been processed yet. Go to the 'Fact Check Claims' page to start.")

def how_it_works_page():
    st.header("🔬 How Our Fact-Checking System Works")
    
    st.markdown("""
    Our misinformation detection system combines cutting-edge technologies to provide 
    real-time fact-checking capabilities:
    """)
    
    # Process flow
    st.subheader("🔄 Processing Pipeline")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 1. **Claim Ingestion**
        - Text preprocessing and normalization
        - Key claim extraction using NLP
        - Pathway streaming data ingestion
        """)
        
        st.markdown("""
        ### 2. **Pathway Processing**
        - Real-time data transformation
        - Vector embedding generation
        - Semantic similarity computation
        """)
        
        st.markdown("""
        ### 3. **RAG-Based Verification**
        - Query vector database for relevant sources
        - Retrieve authoritative fact-check data
        - Cross-reference multiple sources
        """)
    
    with col2:
        st.markdown("""
        ### 4. **AI Analysis**
        - GPT-5 powered claim verification
        - Confidence score calculation
        - Source credibility assessment
        """)
        
        st.markdown("""
        ### 5. **Result Generation**
        - Verification status determination
        - Supporting evidence compilation
        - User-friendly result formatting
        """)
        
        st.markdown("""
        ### 6. **Live Updates**
        - Real-time monitoring dashboard
        - Trend analysis and reporting
        - Performance metrics tracking
        """)
    
    # Technology stack
    st.subheader("🛠️ Technology Stack")
    
    tech_col1, tech_col2, tech_col3 = st.columns(3)
    
    with tech_col1:
        st.markdown("""
        **Core Processing**
        - Pathway (Streaming)
        - Python
        - Pandas/NumPy
        """)
    
    with tech_col2:
        st.markdown("""
        **AI & NLP**
        - OpenAI GPT-5
        - Vector Embeddings
        - NLTK/spaCy
        """)
    
    with tech_col3:
        st.markdown("""
        **Interface**
        - Streamlit
        - Plotly
        - Real-time Updates
        """)
    
    # Accuracy and limitations
    st.subheader("⚠️ Accuracy and Limitations")
    
    st.warning("""
    **Important Disclaimers:**
    - This system provides AI-assisted fact-checking, not definitive truth
    - Results should be verified with additional authoritative sources
    - Confidence scores indicate AI certainty, not absolute accuracy
    - The system may have biases based on training data
    - Always use critical thinking when evaluating claims
    """)

def analytics_page():
    st.header("📈 Analytics Dashboard")
    
    if not st.session_state.monitoring_data:
        st.info("No analytics data available yet. Process some claims first!")
        return
    
    # Create DataFrame from monitoring data
    df = pd.DataFrame(st.session_state.monitoring_data)
    
    # Time series of processed claims
    st.subheader("Claims Processed Over Time")
    df['hour'] = df['timestamp'].dt.floor('H')
    hourly_counts = df.groupby('hour').size().reset_index().rename(columns={0: 'count'})
    
    fig_timeline = px.line(
        hourly_counts, 
        x='hour', 
        y='count',
        title='Claims Processed Per Hour'
    )
    st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Status distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Verification Status Distribution")
        status_counts = df['status'].value_counts()
        fig_pie = px.pie(
            values=status_counts.values, 
            names=status_counts.index,
            title="Status Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("Confidence Score Distribution")
        fig_hist = px.histogram(
            df, 
            x='confidence', 
            title='Confidence Score Distribution',
            nbins=20
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Performance metrics
    st.subheader("Performance Metrics")
    avg_confidence = df['confidence'].mean()
    high_confidence_claims = len(df[df['confidence'] > 0.8])
    low_confidence_claims = len(df[df['confidence'] < 0.5])
    
    met_col1, met_col2, met_col3 = st.columns(3)
    with met_col1:
        st.metric("Average Confidence", f"{avg_confidence:.2f}")
    with met_col2:
        st.metric("High Confidence Claims", high_confidence_claims)
    with met_col3:
        st.metric("Low Confidence Claims", low_confidence_claims)

if __name__ == "__main__":
    main()
