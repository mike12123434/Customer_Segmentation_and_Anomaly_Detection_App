import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.stats import skew
import warnings
import json
from typing import Tuple, Dict, Optional, List
from functools import lru_cache

# Hugging Face Integration
from huggingface_hub import InferenceClient

# Advanced clustering packages check
try:
    from kmodes.kmodes import KModes
    import gower
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="全方位客戶分群 (Advanced Clustering)", 
    layout="wide", 
    page_icon="👥"
)

# ============================================================
# Configuration & Constants
# ============================================================

class Config:
    """Centralized configuration"""
    MAX_GOWER_ROWS = 5000
    SILHOUETTE_SAMPLE_SIZE = 5000
    MIN_SAMPLES_FOR_SILHOUETTE = 50
    CACHE_TTL = 3600
    DEFAULT_RANDOM_STATE = 42
    # Recommended model for structured instruction following
    HF_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3" 

# ============================================================
# Data Processing & Caching
# ============================================================

@st.cache_data(ttl=Config.CACHE_TTL)
def load_data(file) -> pd.DataFrame:
    """Load and validate CSV data with caching"""
    try:
        df = pd.read_csv(file)
        if df.empty:
            raise ValueError("Uploaded file is empty")
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        raise

def detect_column_types(df: pd.DataFrame, features: List[str]) -> Tuple[List[str], List[str]]:
    """Efficiently detect numeric and categorical columns"""
    df_subset = df[features]
    numeric_cols = df_subset.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_subset.select_dtypes(exclude=[np.number]).columns.tolist()
    return numeric_cols, categorical_cols

def smart_preprocessing_numeric(
    data: pd.DataFrame, 
    features: List[str], 
    log_threshold: float = 1.0
) -> Tuple[np.ndarray, Dict, pd.DataFrame]:
    """Optimized preprocessing with better memory management"""
    df_clean = data[features].dropna().copy()
    
    if df_clean.empty:
        raise ValueError("No valid data after removing missing values")
    
    transform_info = {
        'log_features': [],
        'scaler': None,
        'feature_stats': {}
    }
    
    # Process features efficiently
    for feat in features:
        if pd.api.types.is_numeric_dtype(df_clean[feat]):
            col_min = df_clean[feat].min()
            col_skew = skew(df_clean[feat])
            
            transform_info['feature_stats'][feat] = {
                'min': col_min,
                'max': df_clean[feat].max(),
                'skew': col_skew
            }
            
            # Apply log transform only if data is non-negative and highly skewed
            if col_min >= 0 and col_skew > log_threshold:
                df_clean[feat] = np.log1p(df_clean[feat])
                transform_info['log_features'].append(feat)
    
    # Scale data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_clean)
    transform_info['scaler'] = scaler
    
    return scaled_data, transform_info, df_clean

def calculate_multiple_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Calculate multiple clustering quality metrics"""
    mask = labels != -1
    unique_labels = np.unique(labels[mask])
    
    metrics = {
        'silhouette': -1,
        'calinski_harabasz': -1,
        'davies_bouldin': float('inf'),
        'n_clusters': len(unique_labels),
        'noise_ratio': np.sum(labels == -1) / len(labels)
    }
    
    if len(unique_labels) < 2:
        return metrics
    
    X_clean = X[mask]
    labels_clean = labels[mask]
    
    try:
        # Silhouette Score (Sampled for speed if large)
        if len(X_clean) <= Config.SILHOUETTE_SAMPLE_SIZE:
            metrics['silhouette'] = silhouette_score(X_clean, labels_clean)
        else:
            indices = np.random.choice(len(X_clean), Config.SILHOUETTE_SAMPLE_SIZE, replace=False)
            metrics['silhouette'] = silhouette_score(X_clean[indices], labels_clean[indices])
        
        metrics['calinski_harabasz'] = calinski_harabasz_score(X_clean, labels_clean)
        metrics['davies_bouldin'] = davies_bouldin_score(X_clean, labels_clean)
        
    except Exception as e:
        st.warning(f"Error calculating metrics: {e}")
    
    return metrics

def calculate_composite_score(metrics: Dict[str, float]) -> float:
    """Calculate composite score combining multiple metrics"""
    sil_norm = (metrics['silhouette'] + 1) / 2
    noise_penalty = 1 - metrics['noise_ratio']
    db_norm = 1 / (1 + min(metrics['davies_bouldin'], 10))
    
    composite = (
        0.5 * sil_norm +
        0.3 * db_norm +
        0.2 * noise_penalty
    )
    return composite

# ============================================================
# Clustering Algorithms (Optimized)
# ============================================================

def run_kmeans(X: np.ndarray, n_clusters_range: Tuple[int, int]) -> Dict:
    results = {}
    best_score = -1
    best_model = None
    
    progress_bar = st.progress(0)
    st.write(f"⏳ Running K-Means (range: {n_clusters_range})...")
    
    total_iterations = n_clusters_range[1] - n_clusters_range[0] + 1
    
    for idx, k in enumerate(range(n_clusters_range[0], n_clusters_range[1] + 1)):
        model = KMeans(
            n_clusters=k, 
            random_state=Config.DEFAULT_RANDOM_STATE,
            n_init=10,
            max_iter=300
        )
        labels = model.fit_predict(X)
        metrics = calculate_multiple_metrics(X, labels)
        composite_score = calculate_composite_score(metrics)
        
        if composite_score > best_score:
            best_score = composite_score
            best_model = (k, labels, model, metrics)
        
        progress_bar.progress((idx + 1) / total_iterations)
    
    progress_bar.empty()
    
    if best_model:
        results['Best Model'] = {
            'type': 'K-Means',
            'labels': best_model[1],
            'score': best_score,
            'metrics': best_model[3],
            'params': f"k={best_model[0]}",
            'model': best_model[2]
        }
    else:
        results['Best Model'] = {'score': -1, 'labels': [], 'type': 'Failed'}
    
    return results

def run_kmodes(df: pd.DataFrame, n_clusters_range: Tuple[int, int]) -> Dict:
    if not ADVANCED_AVAILABLE:
        st.error("K-Modes not available. Install: pip install kmodes")
        return {'Best Model': {'score': -1, 'type': 'K-Modes Unavailable'}}
    
    results = {}
    best_score = -1
    best_model = None
    
    progress_bar = st.progress(0)
    st.write(f"⏳ Running K-Modes (range: {n_clusters_range})...")
    
    X_matrix = df.values
    total_iterations = n_clusters_range[1] - n_clusters_range[0] + 1
    
    # Pre-encode for metric calculation
    df_encoded = df.copy()
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object':
            df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))
    X_encoded = df_encoded.values
    
    for idx, k in enumerate(range(n_clusters_range[0], n_clusters_range[1] + 1)):
        try:
            km = KModes(
                n_clusters=k, 
                init='Huang', 
                n_init=5, 
                verbose=0, 
                random_state=Config.DEFAULT_RANDOM_STATE
            )
            labels = km.fit_predict(X_matrix)
            metrics = calculate_multiple_metrics(X_encoded, labels)
            composite_score = calculate_composite_score(metrics)
            
            if composite_score > best_score:
                best_score = composite_score
                best_model = (k, labels, km, metrics)
        except Exception as e:
            st.warning(f"K-Modes failed for k={k}: {e}")
            continue
        
        progress_bar.progress((idx + 1) / total_iterations)
    
    progress_bar.empty()
    
    if best_model:
        results['Best Model'] = {
            'type': 'K-Modes',
            'labels': best_model[1],
            'score': best_score,
            'metrics': best_model[3],
            'params': f"k={best_model[0]}",
            'model': best_model[2]
        }
    else:
        results['Best Model'] = {'score': -1, 'type': 'Failed'}
    
    return results

def run_gower_hierarchical(df: pd.DataFrame, n_clusters_range: Tuple[int, int]) -> Dict:
    if not ADVANCED_AVAILABLE:
        st.error("Gower clustering not available. Install: pip install gower")
        return {'Best Model': {'score': -1, 'type': 'Gower Unavailable'}}
    
    results = {}
    rows = df.shape[0]
    
    if rows > Config.MAX_GOWER_ROWS:
        st.error(f"⚠️ Data exceeds {Config.MAX_GOWER_ROWS} rows. Use numeric-only features or sample data.")
        return {'Best Model': {'score': -1, 'type': 'Error: Data too large'}}
    
    st.write("⏳ Computing Gower Distance Matrix...")
    try:
        with st.spinner("Calculating distances..."):
            dist_matrix = gower.gower_matrix(df)
    except Exception as e:
        st.error(f"Gower calculation failed: {e}")
        return {'Best Model': {'score': -1, 'type': 'Error'}}
    
    progress_bar = st.progress(0)
    st.write(f"⏳ Running Hierarchical Clustering (range: {n_clusters_range})...")
    
    best_score = -1
    best_model = None
    total_iterations = n_clusters_range[1] - n_clusters_range[0] + 1
    
    for idx, k in enumerate(range(n_clusters_range[0], n_clusters_range[1] + 1)):
        try:
            model = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='average')
            labels = model.fit_predict(dist_matrix)
            
            mask = labels != -1
            unique_labels = np.unique(labels[mask])
            metrics = {'silhouette': -1, 'n_clusters': len(unique_labels), 'noise_ratio': 0}
            
            if len(unique_labels) >= 2:
                dist_clean = dist_matrix[mask][:, mask]
                labels_clean = labels[mask]
                if len(dist_clean) <= Config.SILHOUETTE_SAMPLE_SIZE:
                    metrics['silhouette'] = silhouette_score(dist_clean, labels_clean, metric='precomputed')
                else:
                    metrics['silhouette'] = 0
            
            composite_score = (metrics['silhouette'] + 1) / 2
            
            if composite_score > best_score:
                best_score = composite_score
                best_model = (k, labels, metrics)
        except Exception:
            continue
        
        progress_bar.progress((idx + 1) / total_iterations)
    
    progress_bar.empty()
    
    if best_model:
        results['Best Model'] = {
            'type': 'Gower + Hierarchical',
            'labels': best_model[1],
            'score': best_score,
            'metrics': best_model[2],
            'params': f"k={best_model[0]}",
            'model': None
        }
    else:
        results['Best Model'] = {'score': -1, 'type': 'Failed'}
    
    return results

# ============================================================
# Anomaly Detection (Optimized)
# ============================================================

def run_anomaly_detection(df: pd.DataFrame, features: List[str], contamination: float = 0.05) -> Tuple[pd.DataFrame, np.ndarray]:
    """Optimized Isolation Forest with better preprocessing"""
    st.write("🔍 Running Isolation Forest anomaly detection...")
    
    df_working = df[features].copy()
    
    for col in df_working.columns:
        if not pd.api.types.is_numeric_dtype(df_working[col]):
            le = LabelEncoder()
            df_working[col] = le.fit_transform(df_working[col].astype(str))
    
    df_working = df_working.fillna(df_working.median())
    
    model = IsolationForest(
        contamination=contamination, 
        random_state=Config.DEFAULT_RANDOM_STATE,
        n_estimators=100,
        n_jobs=-1
    )
    
    predictions = model.fit_predict(df_working)
    anomaly_scores = model.score_samples(df_working)
    
    anomalies = df[predictions == -1].copy()
    anomalies['anomaly_score'] = anomaly_scores[predictions == -1]
    
    return anomalies, predictions

# ============================================================
# Hugging Face LLM Integration
# ============================================================

def query_huggingface(client: InferenceClient, prompt: str, max_tokens: int = 1000) -> str:
    """Helper to query HF Inference API"""
    try:
        response = client.text_generation(
            prompt=prompt,
            model=Config.HF_MODEL_ID,
            max_new_tokens=max_tokens,
            temperature=0.7,
            return_full_text=False
        )
        return response.strip()
    except Exception as e:
        raise Exception(f"Hugging Face API Error: {str(e)}")

def generate_cluster_descriptions(
    df_viz: pd.DataFrame, 
    features: List[str], 
    overall_means: pd.Series,
    hf_token: Optional[str] = None,
    threshold: float = 0.15
) -> Tuple[Dict, pd.DataFrame]:
    """
    Generate Cluster Statistics and AI-powered descriptions using Hugging Face.
    """
    desc_text = {}
    grouped = df_viz[df_viz['Cluster'] != '-1'].groupby('Cluster')
    
    # 1. Calculate Cluster Statistics
    stats_data = []
    for cluster_id, cluster_data in grouped:
        row = {'Cluster': cluster_id}
        for feat in features:
            if pd.api.types.is_numeric_dtype(df_viz[feat]):
                row[feat] = round(cluster_data[feat].mean(), 2)
            else:
                modes = cluster_data[feat].mode()
                row[feat] = modes.iloc[0] if not modes.empty else "N/A"
        stats_data.append(row)
    
    cluster_stats_df = pd.DataFrame(stats_data)
    if not cluster_stats_df.empty:
        cluster_stats_df = cluster_stats_df.set_index('Cluster').sort_index()

    # 2. AI Descriptions
    if not hf_token:
        for cluster_id in grouped.groups.keys():
            desc_text[str(cluster_id)] = {
                "輪廓": "未啟用 AI 分析",
                "特徵": "請提供 Hugging Face Token 以獲得詳細解讀",
                "策略": "觀察上方視覺化圖表進行人工分析"
            }
        return desc_text, cluster_stats_df
    
    try:
        client = InferenceClient(token=hf_token)
        
        for cluster_id, cluster_data in grouped:
            cluster_size = len(cluster_data)
            cluster_pct = (cluster_size / len(df_viz)) * 100
            
            feature_summary = []
            for feat in features:
                if pd.api.types.is_numeric_dtype(df_viz[feat]):
                    cluster_val = cluster_data[feat].mean()
                    overall_val = overall_means.get(feat, 0)
                    if overall_val != 0:
                        diff_pct = ((cluster_val - overall_val) / overall_val) * 100
                        if abs(diff_pct) > threshold * 100:
                            direction = "高於" if diff_pct > 0 else "低於"
                            feature_summary.append(f"{feat}: {cluster_val:.2f} ({direction}平均 {abs(diff_pct):.1f}%)")
                else:
                    mode_val = cluster_data[feat].mode()
                    if len(mode_val) > 0:
                        feature_summary.append(f"{feat}: 主要為 {mode_val.iloc[0]}")
            
            # Formulate prompt for open source models (Mistral instruction format)
            prompt = f"""[INST] You are a data analysis expert. Analyze the following customer cluster data and provide insights in Traditional Chinese (繁體中文).

Cluster ID: {cluster_id}
Size: {cluster_size} ({cluster_pct:.1f}%)
Key Features:
{chr(10).join(feature_summary)}

Return valid JSON strictly with the following keys: "輪廓" (profile), "特徵" (unique traits), "策略" (strategy). Do not include markdown code blocks.
Example format: {{"輪廓": "...", "特徵": "...", "策略": "..."}} [/INST]"""
            
            try:
                response_text = query_huggingface(client, prompt)
                # Cleanup JSON string
                if "```" in response_text:
                    response_text = response_text.split("```")[1].replace("json", "").strip()
                
                desc_text[str(cluster_id)] = json.loads(response_text)
            except json.JSONDecodeError:
                desc_text[str(cluster_id)] = {
                    "輪廓": response_text[:200],
                    "特徵": "解析 JSON 失敗，顯示原始回應",
                    "策略": "請參考數據表"
                }
            except Exception as e:
                st.warning(f"Error for cluster {cluster_id}: {e}")
                desc_text[str(cluster_id)] = {"輪廓": "分析失敗", "特徵": "API 請求錯誤", "策略": "無"}
                
    except Exception as e:
        st.error(f"Hugging Face Client Error: {e}")
        for cluster_id in grouped.groups.keys():
            desc_text[str(cluster_id)] = {"輪廓": "Client Error", "特徵": "Init failed", "策略": "Check Token"}
    
    return desc_text, cluster_stats_df

def generate_anomaly_insights(
    anomalies: pd.DataFrame,
    features: List[str],
    hf_token: Optional[str] = None
) -> str:
    """Generate AI-powered insights about detected anomalies via Hugging Face"""
    if not hf_token or anomalies.empty:
        return "未啟用 AI 分析或無異常資料"
    
    try:
        client = InferenceClient(token=hf_token)
        sample_size = min(5, len(anomalies))
        
        anomaly_stats = []
        for feat in features:
            if pd.api.types.is_numeric_dtype(anomalies[feat]):
                anomaly_stats.append(f"{feat}: range {anomalies[feat].min():.2f}-{anomalies[feat].max():.2f}")
        
        prompt = f"""[INST] You are an anomaly detection expert. Analyze these anomalies detected in a dataset.
Count: {len(anomalies)}
Stats: {'; '.join(anomaly_stats)}

Provide a summary in Traditional Chinese (繁體中文) answering:
1. What might these anomalies represent?
2. Are they critical?
3. Recommended actions.
Keep it concise (max 150 words). [/INST]"""
        
        return query_huggingface(client, prompt, max_tokens=500)
        
    except Exception as e:
        return f"AI 分析失敗: {e}"

# ============================================================
# Visualization
# ============================================================

def create_cluster_visualization(df: pd.DataFrame, features: List[str], labels: np.ndarray, title: str = "Results") -> None:
    df_viz = df.copy()
    df_viz['Cluster'] = labels.astype(str)
    df_viz_clean = df_viz[df_viz['Cluster'] != '-1']
    plot_features = [f for f in features if f in df_viz_clean.columns]
    
    if len(plot_features) < 2:
        st.warning("Need at least 2 valid features for visualization")
        return
    
    try:
        if len(plot_features) == 2:
            fig = px.scatter(
                df_viz_clean, x=plot_features[0], y=plot_features[1], 
                color='Cluster', title=title, template='plotly_white',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_traces(marker=dict(size=8, opacity=0.7))
        else:
            fig = px.scatter_3d(
                df_viz_clean, x=plot_features[0], y=plot_features[1], z=plot_features[2],
                color='Cluster', title=title, template='plotly_white',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_traces(marker=dict(size=5, opacity=0.7))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Visualization error: {e}")

# ============================================================
# Main Application
# ============================================================

def main():
    st.title("👥 全方位客戶分群系統 (Advanced Clustering)")
    st.markdown("### 智能客戶分群與異常偵測系統")
    
    with st.sidebar:
        st.header("📁 1. Upload Data")
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        st.divider()
        st.header("⚙️ 2. Clustering Parameters")
        min_c = st.number_input("Minimum clusters", 2, 5, 2)
        max_c = st.number_input("Maximum clusters", 6, 20, 8)
        
        st.divider()
        st.header("🔍 3. Anomaly Detection")
        contamination = st.slider("Contamination ratio", 0.01, 0.20, 0.05, 0.01)
        
        st.divider()
        st.header("🤖 4. AI Integration")
        hf_token = st.text_input("Hugging Face Access Token", type="password")
        if hf_token:
             st.caption("✅ Token received. AI Insights enabled.")
    
    if not uploaded_file:
        st.info("👆 Please upload a CSV file to begin")
        return
    
    try:
        df = load_data(uploaded_file)
    except Exception:
        return
    
    st.subheader("📊 Data Preview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())
    
    with st.expander("View raw data"):
        st.dataframe(df.head(20))
    
    # Feature Selection
    st.subheader("🎯 Feature Selection")
    all_cols = df.columns.tolist()
    potential_cols = [c for c in all_cols if 'ID' not in c.upper() and df[c].nunique() < len(df)]
    
    selected_features = st.multiselect(
        "Select 2-3 features for clustering:", potential_cols, max_selections=3
    )
    
    if len(selected_features) < 2:
        st.warning("⚠️ Please select at least 2 features")
        return
    
    # Logic Determination
    num_cols, cat_cols = detect_column_types(df, selected_features)
    if len(cat_cols) == 0:
        algo_type, algo_color = "Numeric (K-Means)", "🔵"
    elif len(num_cols) == 0:
        algo_type, algo_color = "Categorical (K-Modes)", "🟢"
    else:
        algo_type, algo_color = "Mixed (Gower + Hierarchical)", "🟣"
    
    st.info(f"{algo_color} **Detected algorithm:** {algo_type}")
    
    # Execution Buttons
    col_cluster, col_anomaly = st.columns(2)
    with col_cluster:
        run_clustering = st.button("🚀 Start Clustering", type="primary", use_container_width=True)
    with col_anomaly:
        run_anomaly = st.button("🔍 Detect Anomalies", use_container_width=True)
    
    # Clustering Logic
    if run_clustering:
        with st.spinner("🔄 Processing..."):
            try:
                df_used = df[selected_features].dropna().copy()
                st.session_state.features = selected_features
                st.session_state.df_used = df_used
                
                if algo_type == "Numeric (K-Means)":
                    X_scaled, _, _ = smart_preprocessing_numeric(df, selected_features)
                    results = run_kmeans(X_scaled, (min_c, max_c))
                elif algo_type == "Categorical (K-Modes)":
                    results = run_kmodes(df_used, (min_c, max_c))
                else:
                    results = run_gower_hierarchical(df_used, (min_c, max_c))
                
                st.session_state.results = results
                st.session_state.ran_clustering = True
            except Exception as e:
                st.error(f"Clustering error: {e}")

    # Display Cluster Results
    if st.session_state.get('ran_clustering'):
        st.divider()
        st.header("🎯 Clustering Results")
        best_res = st.session_state.results.get('Best Model')
        
        if best_res and best_res['score'] > 0:
            metrics = best_res.get('metrics', {})
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Model", best_res['type'])
            c2.metric("Score", f"{best_res['score']:.4f}")
            c3.metric("Silhouette", f"{metrics.get('silhouette', -1):.4f}")
            c4.metric("Clusters", metrics.get('n_clusters', 0))
            
            create_cluster_visualization(st.session_state.df_used, selected_features, best_res['labels'], best_res['type'])
            
            # Statistics & AI Analysis
            st.divider()
            st.subheader("📋 Insights & Statistics")
            
            numeric_features = [f for f in selected_features if pd.api.types.is_numeric_dtype(df[f])]
            overall_means = df[numeric_features].mean() if numeric_features else pd.Series()
            
            descriptions, stats_df = generate_cluster_descriptions(
                st.session_state.df_used.assign(Cluster=best_res['labels'].astype(str)), 
                selected_features, overall_means, hf_token
            )
            
            st.dataframe(stats_df.style.background_gradient(cmap='Blues'), use_container_width=True)
            
            if hf_token:
                st.subheader("💡 AI Business Insights")
                for cluster_id, desc in descriptions.items():
                    if cluster_id != '-1':
                        with st.expander(f"📊 Cluster {cluster_id} Analysis", expanded=True):
                            c_a, c_b, c_c = st.columns(3)
                            c_a.markdown(f"**👤 Profile**\n{desc.get('輪廓', 'N/A')}")
                            c_b.markdown(f"**🔍 Traits**\n{desc.get('特徵', 'N/A')}")
                            c_c.markdown(f"**💼 Strategy**\n{desc.get('策略', 'N/A')}")
            else:
                st.info("💡 Enter Hugging Face Token in sidebar to enable AI analysis.")

    # Anomaly Logic
    if run_anomaly:
        with st.spinner("🔍 Detecting..."):
            anomalies, predictions = run_anomaly_detection(df, selected_features, contamination)
            st.session_state.anomalies = anomalies
            st.session_state.ran_anomaly = True
            
    if st.session_state.get('ran_anomaly'):
        st.divider()
        st.header("⚠️ Anomaly Results")
        anomalies = st.session_state.anomalies
        st.metric("Detected Anomalies", len(anomalies))
        st.dataframe(anomalies.head(20))
        
        if hf_token:
            st.markdown("### 🤖 AI Anomaly Analysis")
            st.info(generate_anomaly_insights(anomalies, selected_features, hf_token))

if __name__ == "__main__":
    if 'ran_clustering' not in st.session_state: st.session_state.ran_clustering = False
    if 'ran_anomaly' not in st.session_state: st.session_state.ran_anomaly = False
    main()
