"""
Customer Segmentation & Anomaly Detection System
Minimal, efficient, practical design with optional Gemini AI insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings
import json
import os

# Optional Gemini integration
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Customer Segmentation", 
    layout="wide"
)

# ============================================================
# Configuration
# ============================================================

class Config:
    MAX_ROWS_HIERARCHICAL = 3000
    SILHOUETTE_SAMPLE_SIZE = 3000
    DEFAULT_RANDOM_STATE = 42
    CACHE_TTL = 3600

# ============================================================
# Data Processing
# ============================================================

@st.cache_data(ttl=Config.CACHE_TTL)
def load_data(file) -> pd.DataFrame:
    """Load and validate CSV data"""
    try:
        df = pd.read_csv(file)
        if df.empty:
            raise ValueError("File is empty")
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        raise

def preprocess_data(df: pd.DataFrame, features: list) -> tuple:
    """Preprocess data for clustering"""
    df_clean = df[features].copy()
    
    # Handle missing values
    for col in df_clean.columns:
        if df_clean[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            else:
                df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown', inplace=True)
    
    # Encode categorical variables
    encoders = {}
    for col in df_clean.columns:
        if not pd.api.types.is_numeric_dtype(df_clean[col]):
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))
            encoders[col] = le
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)
    
    return X_scaled, df_clean, scaler, encoders

def calculate_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    """Calculate clustering quality metrics"""
    mask = labels != -1
    unique_labels = np.unique(labels[mask])
    
    metrics = {
        'n_clusters': len(unique_labels),
        'silhouette': -1,
        'davies_bouldin': float('inf'),
        'noise_ratio': np.sum(labels == -1) / len(labels)
    }
    
    if len(unique_labels) < 2 or np.sum(mask) < 10:
        return metrics
    
    X_clean = X[mask]
    labels_clean = labels[mask]
    
    try:
        # Sample for large datasets
        if len(X_clean) > Config.SILHOUETTE_SAMPLE_SIZE:
            indices = np.random.choice(len(X_clean), Config.SILHOUETTE_SAMPLE_SIZE, replace=False)
            metrics['silhouette'] = silhouette_score(X_clean[indices], labels_clean[indices])
        else:
            metrics['silhouette'] = silhouette_score(X_clean, labels_clean)
        
        metrics['davies_bouldin'] = davies_bouldin_score(X_clean, labels_clean)
    except Exception:
        pass
    
    return metrics

# ============================================================
# Clustering Algorithms
# ============================================================

def run_kmeans(X: np.ndarray, n_range: tuple) -> dict:
    """Run K-Means clustering with optimal k selection"""
    best_score = -1
    best_result = None
    
    progress_bar = st.progress(0)
    status = st.empty()
    
    total = n_range[1] - n_range[0] + 1
    
    for idx, k in enumerate(range(n_range[0], n_range[1] + 1)):
        status.text(f"Testing K-Means with k={k}...")
        
        model = KMeans(
            n_clusters=k,
            random_state=Config.DEFAULT_RANDOM_STATE,
            n_init=10,
            max_iter=300
        )
        labels = model.fit_predict(X)
        metrics = calculate_metrics(X, labels)
        
        # Composite score: silhouette (normalized) + noise penalty
        score = (metrics['silhouette'] + 1) / 2 * (1 - metrics['noise_ratio'])
        
        if score > best_score:
            best_score = score
            best_result = {
                'type': 'K-Means',
                'k': k,
                'labels': labels,
                'metrics': metrics,
                'score': score,
                'model': model
            }
        
        progress_bar.progress((idx + 1) / total)
    
    progress_bar.empty()
    status.empty()
    
    return best_result

def run_hierarchical(X: np.ndarray, n_range: tuple) -> dict:
    """Run Hierarchical clustering"""
    if len(X) > Config.MAX_ROWS_HIERARCHICAL:
        st.warning(f"Dataset too large for hierarchical clustering (>{Config.MAX_ROWS_HIERARCHICAL} rows). Skipping.")
        return None
    
    best_score = -1
    best_result = None
    
    progress_bar = st.progress(0)
    status = st.empty()
    
    total = n_range[1] - n_range[0] + 1
    
    for idx, k in enumerate(range(n_range[0], n_range[1] + 1)):
        status.text(f"Testing Hierarchical with k={k}...")
        
        model = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels = model.fit_predict(X)
        metrics = calculate_metrics(X, labels)
        
        score = (metrics['silhouette'] + 1) / 2
        
        if score > best_score:
            best_score = score
            best_result = {
                'type': 'Hierarchical',
                'k': k,
                'labels': labels,
                'metrics': metrics,
                'score': score,
                'model': model
            }
        
        progress_bar.progress((idx + 1) / total)
    
    progress_bar.empty()
    status.empty()
    
    return best_result

# ============================================================
# Anomaly Detection
# ============================================================

def detect_anomalies(X: np.ndarray, contamination: float = 0.05) -> tuple:
    """Detect anomalies using Isolation Forest"""
    model = IsolationForest(
        contamination=contamination,
        random_state=Config.DEFAULT_RANDOM_STATE,
        n_estimators=100,
        max_samples='auto',
        n_jobs=-1
    )
    
    predictions = model.fit_predict(X)
    scores = model.score_samples(X)
    
    return predictions, scores

# ============================================================
# Gemini AI Integration (Optional)
# ============================================================

def generate_cluster_insights(
    df: pd.DataFrame,
    features: list,
    labels: np.ndarray,
    api_key: str = None
) -> dict:
    """Generate AI-powered cluster insights using Gemini"""
    
    insights = {}
    df_viz = df.copy()
    df_viz['Cluster'] = labels.astype(str)
    
    # Calculate cluster statistics
    grouped = df_viz[df_viz['Cluster'] != '-1'].groupby('Cluster')
    
    for cluster_id, cluster_data in grouped:
        # Basic statistics
        cluster_size = len(cluster_data)
        cluster_pct = (cluster_size / len(df_viz)) * 100
        
        # Feature summary
        feature_summary = []
        for feat in features:
            if pd.api.types.is_numeric_dtype(df[feat]):
                mean_val = cluster_data[feat].mean()
                overall_mean = df[feat].mean()
                diff_pct = ((mean_val - overall_mean) / overall_mean * 100) if overall_mean != 0 else 0
                
                if abs(diff_pct) > 10:
                    direction = "higher" if diff_pct > 0 else "lower"
                    feature_summary.append(f"{feat}: {mean_val:.2f} ({direction} than average by {abs(diff_pct):.1f}%)")
            else:
                mode_val = cluster_data[feat].mode()
                if len(mode_val) > 0:
                    feature_summary.append(f"{feat}: predominantly {mode_val.iloc[0]}")
        
        # Default insight (no AI)
        insights[str(cluster_id)] = {
            'size': f"{cluster_size} customers ({cluster_pct:.1f}%)",
            'profile': ', '.join(feature_summary) if feature_summary else 'No significant patterns',
            'recommendation': 'Manual analysis recommended'
        }
        
        # Generate AI insights if API key provided
        if api_key and GEMINI_AVAILABLE:
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                prompt = f"""
Analyze this customer segment:

Cluster {cluster_id}:
- Size: {cluster_size} customers ({cluster_pct:.1f}%)
- Features: {', '.join(feature_summary)}

Provide a concise analysis in JSON format with these keys:
- "profile": Brief customer profile (2-3 sentences)
- "recommendation": Marketing or business recommendation (2-3 sentences)

Keep it practical and business-focused.
"""
                
                response = model.generate_content(prompt)
                text = response.text.strip()
                
                # Clean JSON response
                if '```json' in text:
                    text = text.split('```json')[1].split('```')[0].strip()
                elif '```' in text:
                    text = text.split('```')[1].strip()
                
                ai_insight = json.loads(text)
                insights[str(cluster_id)].update({
                    'profile': ai_insight.get('profile', insights[str(cluster_id)]['profile']),
                    'recommendation': ai_insight.get('recommendation', insights[str(cluster_id)]['recommendation'])
                })
                
            except Exception as e:
                st.warning(f"AI analysis failed for cluster {cluster_id}: {str(e)}")
    
    return insights

def generate_anomaly_insights(
    df: pd.DataFrame,
    features: list,
    anomaly_mask: np.ndarray,
    api_key: str = None
) -> str:
    """Generate AI insights about anomalies"""
    
    anomalies = df[anomaly_mask == -1]
    n_anomalies = len(anomalies)
    anomaly_rate = (n_anomalies / len(df)) * 100
    
    # Basic statistics
    stats = []
    for feat in features:
        if pd.api.types.is_numeric_dtype(df[feat]):
            anom_mean = anomalies[feat].mean()
            normal_mean = df[anomaly_mask == 1][feat].mean()
            stats.append(f"{feat}: anomaly avg {anom_mean:.2f} vs normal avg {normal_mean:.2f}")
    
    basic_insight = f"Detected {n_anomalies} anomalies ({anomaly_rate:.2f}%). Key differences: {', '.join(stats[:3])}"
    
    if not api_key or not GEMINI_AVAILABLE:
        return basic_insight
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
Analyze these anomalies in customer data:

- Total anomalies: {n_anomalies} ({anomaly_rate:.2f}%)
- Key differences: {', '.join(stats)}

Provide:
1. What these anomalies might represent (business perspective)
2. Whether they need attention and why
3. Recommended actions

Keep response concise (3-5 sentences).
"""
        
        response = model.generate_content(prompt)
        return response.text.strip()
        
    except Exception as e:
        return f"{basic_insight} (AI analysis unavailable: {str(e)})"

# ============================================================
# Visualization
# ============================================================

def plot_clusters(df: pd.DataFrame, features: list, labels: np.ndarray):
    """Create cluster visualization"""
    df_viz = df.copy()
    df_viz['Cluster'] = labels.astype(str)
    df_clean = df_viz[df_viz['Cluster'] != '-1']
    
    if len(features) < 2 or df_clean.empty:
        st.warning("Insufficient data for visualization")
        return
    
    if len(features) == 2:
        fig = px.scatter(
            df_clean,
            x=features[0],
            y=features[1],
            color='Cluster',
            title="Cluster Distribution",
            template='plotly_white'
        )
    else:
        fig = px.scatter_3d(
            df_clean,
            x=features[0],
            y=features[1],
            z=features[2],
            color='Cluster',
            title="Cluster Distribution (3D)",
            template='plotly_white'
        )
    
    fig.update_traces(marker=dict(size=6, opacity=0.7))
    st.plotly_chart(fig, use_container_width=True)

def plot_anomalies(df: pd.DataFrame, features: list, predictions: np.ndarray):
    """Create anomaly visualization"""
    df_viz = df.copy()
    df_viz['Type'] = np.where(predictions == -1, 'Anomaly', 'Normal')
    
    if len(features) < 2:
        st.warning("Need at least 2 features for visualization")
        return
    
    if len(features) == 2:
        fig = px.scatter(
            df_viz,
            x=features[0],
            y=features[1],
            color='Type',
            color_discrete_map={'Normal': '#CCCCCC', 'Anomaly': '#FF4444'},
            title="Anomaly Detection Results",
            template='plotly_white'
        )
    else:
        fig = px.scatter_3d(
            df_viz,
            x=features[0],
            y=features[1],
            z=features[2],
            color='Type',
            color_discrete_map={'Normal': '#CCCCCC', 'Anomaly': '#FF4444'},
            title="Anomaly Detection Results (3D)",
            template='plotly_white'
        )
    
    fig.update_traces(marker=dict(size=6, opacity=0.7))
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Main Application
# ============================================================

def main():
    st.title("Customer Segmentation & Anomaly Detection")
    st.caption("Efficient clustering and outlier detection for customer analysis")
    
    # Sidebar
    with st.sidebar:
        st.header("1. Upload Data")
        uploaded_file = st.file_uploader("CSV file", type=['csv'])
        
        st.divider()
        st.header("2. Parameters")
        
        min_clusters = st.number_input("Min clusters", 2, 5, 2)
        max_clusters = st.number_input("Max clusters", 6, 20, 6)
        
        contamination = st.slider(
            "Anomaly rate",
            0.01, 0.20, 0.05, 0.01,
            help="Expected proportion of anomalies"
        )
        
        st.divider()
        st.header("3. AI Insights (Optional)")
        
        if GEMINI_AVAILABLE:
            gemini_key = st.text_input("Gemini API Key", type="password")
            if not gemini_key:
                env_key = os.environ.get("GEMINI_API_KEY")
                if env_key:
                    gemini_key = env_key
                    st.caption("Using API key from environment")
        else:
            st.warning("Install google-generativeai for AI insights")
            gemini_key = None
    
    if not uploaded_file:
        st.info("Upload a CSV file to begin")
        return
    
    # Load data
    try:
        df = load_data(uploaded_file)
    except:
        return
    
    # Data overview
    st.subheader("Data Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing", df.isnull().sum().sum())
    
    with st.expander("View data"):
        st.dataframe(df.head(20), use_container_width=True)
    
    # Feature selection
    st.subheader("Feature Selection")
    
    # Filter suitable columns
    suitable_cols = []
    for col in df.columns:
        # Skip likely ID columns
        if col.upper().endswith('ID') or 'ID' in col.upper():
            continue
        
        # Skip high-cardinality categorical columns
        if df[col].dtype == 'object':
            if df[col].nunique() > min(50, len(df) * 0.5):
                continue
        
        suitable_cols.append(col)
    
    selected_features = st.multiselect(
        "Select 2-3 features",
        suitable_cols,
        max_selections=3,
        help="Choose features that represent customer behavior"
    )
    
    if len(selected_features) < 2:
        st.warning("Select at least 2 features")
        return
    
    # Show feature info
    with st.expander("Feature information"):
        info_df = pd.DataFrame({
            'Feature': selected_features,
            'Type': [str(df[f].dtype) for f in selected_features],
            'Unique': [df[f].nunique() for f in selected_features],
            'Missing': [df[f].isnull().sum() for f in selected_features]
        })
        st.dataframe(info_df, use_container_width=True)
    
    # Action buttons
    col1, col2 = st.columns(2)
    
    with col1:
        run_clustering = st.button("Run Clustering", type="primary", use_container_width=True)
    
    with col2:
        run_anomaly = st.button("Detect Anomalies", use_container_width=True)
    
    # Clustering
    if run_clustering:
        with st.spinner("Running clustering algorithms..."):
            try:
                # Preprocess
                X_scaled, df_clean, scaler, encoders = preprocess_data(df, selected_features)
                
                st.session_state.X_scaled = X_scaled
                st.session_state.df_clean = df_clean
                st.session_state.selected_features = selected_features
                
                # Run algorithms
                st.write("Testing K-Means...")
                kmeans_result = run_kmeans(X_scaled, (min_clusters, max_clusters))
                
                hierarchical_result = None
                if len(X_scaled) <= Config.MAX_ROWS_HIERARCHICAL:
                    st.write("Testing Hierarchical...")
                    hierarchical_result = run_hierarchical(X_scaled, (min_clusters, max_clusters))
                
                # Select best
                results = [r for r in [kmeans_result, hierarchical_result] if r is not None]
                if not results:
                    st.error("Clustering failed")
                    return
                
                best_result = max(results, key=lambda x: x['score'])
                st.session_state.cluster_result = best_result
                st.session_state.ran_clustering = True
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                return
    
    # Display clustering results
    if st.session_state.get('ran_clustering'):
        st.divider()
        st.header("Clustering Results")
        
        result = st.session_state.cluster_result
        metrics = result['metrics']
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Algorithm", result['type'])
        col2.metric("Clusters", result['k'])
        col3.metric("Silhouette", f"{metrics['silhouette']:.3f}")
        col4.metric("Score", f"{result['score']:.3f}")
        
        # Visualization
        st.subheader("Cluster Visualization")
        plot_clusters(
            st.session_state.df_clean,
            st.session_state.selected_features,
            result['labels']
        )
        
        # Statistics
        st.subheader("Cluster Statistics")
        df_viz = st.session_state.df_clean.copy()
        df_viz['Cluster'] = result['labels'].astype(str)
        
        stats = df_viz['Cluster'].value_counts().reset_index()
        stats.columns = ['Cluster', 'Count']
        stats['Percentage'] = (stats['Count'] / len(df_viz) * 100).round(2)
        st.dataframe(stats, use_container_width=True)
        
        # Cluster profiles
        st.subheader("Cluster Profiles")
        grouped = df_viz[df_viz['Cluster'] != '-1'].groupby('Cluster')
        
        profile_data = []
        for cluster_id, cluster_data in grouped:
            row = {'Cluster': cluster_id}
            for feat in st.session_state.selected_features:
                if pd.api.types.is_numeric_dtype(df_viz[feat]):
                    row[feat] = round(cluster_data[feat].mean(), 2)
                else:
                    modes = cluster_data[feat].mode()
                    row[feat] = modes.iloc[0] if not modes.empty else "N/A"
            profile_data.append(row)
        
        profile_df = pd.DataFrame(profile_data).set_index('Cluster')
        st.dataframe(profile_df, use_container_width=True)
        
        # AI Insights
        if gemini_key:
            st.subheader("AI-Powered Insights")
            with st.spinner("Generating insights..."):
                insights = generate_cluster_insights(
                    df,
                    st.session_state.selected_features,
                    result['labels'],
                    gemini_key
                )
            
            for cluster_id, insight in insights.items():
                if cluster_id != '-1':
                    with st.expander(f"Cluster {cluster_id}", expanded=True):
                        st.write(f"**Size:** {insight['size']}")
                        st.write(f"**Profile:** {insight['profile']}")
                        st.write(f"**Recommendation:** {insight['recommendation']}")
        else:
            st.info("Provide Gemini API key for AI-powered insights")
    
    # Anomaly Detection
    if run_anomaly:
        with st.spinner("Detecting anomalies..."):
            try:
                if 'X_scaled' not in st.session_state:
                    X_scaled, df_clean, _, _ = preprocess_data(df, selected_features)
                    st.session_state.X_scaled = X_scaled
                    st.session_state.df_clean = df_clean
                    st.session_state.selected_features = selected_features
                
                predictions, scores = detect_anomalies(
                    st.session_state.X_scaled,
                    contamination
                )
                
                st.session_state.anomaly_predictions = predictions
                st.session_state.anomaly_scores = scores
                st.session_state.ran_anomaly = True
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Display anomaly results
    if st.session_state.get('ran_anomaly'):
        st.divider()
        st.header("Anomaly Detection Results")
        
        predictions = st.session_state.anomaly_predictions
        n_anomalies = np.sum(predictions == -1)
        anomaly_rate = (n_anomalies / len(predictions)) * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Anomalies", n_anomalies)
        col2.metric("Rate", f"{anomaly_rate:.2f}%")
        col3.metric("Normal", np.sum(predictions == 1))
        
        if n_anomalies > 0:
            # Show anomalies
            st.subheader("Detected Anomalies")
            anomaly_df = df[predictions == -1].copy()
            anomaly_df['anomaly_score'] = st.session_state.anomaly_scores[predictions == -1]
            st.dataframe(anomaly_df.head(20), use_container_width=True)
            
            # Visualization
            st.subheader("Anomaly Visualization")
            plot_anomalies(
                st.session_state.df_clean,
                st.session_state.selected_features,
                predictions
            )
            
            # AI Insights
            if gemini_key:
                st.subheader("AI Analysis")
                with st.spinner("Analyzing anomalies..."):
                    insight = generate_anomaly_insights(
                        df,
                        st.session_state.selected_features,
                        predictions,
                        gemini_key
                    )
                st.info(insight)
            else:
                st.info("Provide Gemini API key for AI-powered analysis")
        else:
            st.success("No anomalies detected with current settings")

if __name__ == "__main__":
    # Initialize session state
    if 'ran_clustering' not in st.session_state:
        st.session_state.ran_clustering = False
    if 'ran_anomaly' not in st.session_state:
        st.session_state.ran_anomaly = False
    
    main()