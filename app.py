import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(page_title="Telco Churn Predictor", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3em;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 15px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    /* Sidebar Customization */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: white;
    }
    .sidebar-title {
        color: white;
        font-size: 1.5em;
        font-weight: bold;
        margin-bottom: 20px;
    }
    /* Navigation Radio Buttons */
    [data-testid="stSidebar"] .stRadio > label {
        font-size: 1.1em !important;
        padding: 10px !important;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    [data-testid="stSidebar"] .stRadio > div {
        gap: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load('churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    if isinstance(scaler, list):  # Old format compatibility
        columns = scaler
        scaler = joblib.load('scaler.pkl')
    else:
        columns = joblib.load('model_columns.pkl')
    return model, scaler, columns

model, scaler, columns = load_model_and_scaler()

# Load dataset for analysis
@st.cache_data
def load_data():
    df = pd.read_excel('Telco_customer_churn.xlsx')
    # Drop unnecessary columns (same as in notebook)
    cols_to_drop = [
        'CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code',
        'Lat Long', 'Latitude', 'Longitude', 'Churn Label', 'Churn Score',
        'CLTV', 'Churn Reason'
    ]
    return df.drop(columns=cols_to_drop)

df = load_data()

# Helper function to preprocess user input for prediction
@st.cache_data
def preprocess_raw_data(df_raw):
    """Preprocess raw dataframe the same way as training data"""
    # Make a copy to avoid modifying the original
    df_processed = df_raw.copy()
    
    # Strip whitespace from all string columns
    string_cols = df_processed.select_dtypes(include=['object']).columns
    for col in string_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].str.strip()
    
    # Convert Total Charges to numeric (this may create NaNs if values are non-numeric)
    if 'Total Charges' in df_processed.columns:
        df_processed['Total Charges'] = pd.to_numeric(df_processed['Total Charges'], errors='coerce')
    
    # Drop rows with NaN values in numeric columns
    numeric_cols_present = ['Total Charges', 'Tenure Months', 'Monthly Charges']
    numeric_cols_present = [col for col in numeric_cols_present if col in df_processed.columns]
    
    if numeric_cols_present:
        df_processed = df_processed.dropna(subset=numeric_cols_present)
    
    # Check if we have any data left
    if len(df_processed) == 0:
        st.warning("⚠️ No valid data after filtering. Original data may have had format issues.")
        return None
    
    # Convert string columns to category type
    obj_cols = df_processed.select_dtypes(include=['object']).columns
    if len(obj_cols) > 0:
        df_processed[obj_cols] = df_processed[obj_cols].astype('category')
    
    # One-hot encode categorical columns
    df_encoded = pd.get_dummies(df_processed, columns=obj_cols, drop_first=True)
    
    # Scale numerical columns
    numerical_cols = ['Tenure Months', 'Monthly Charges', 'Total Charges']
    numerical_cols = [col for col in numerical_cols if col in df_encoded.columns]
    
    if numerical_cols and len(df_encoded) > 0:
        scaler_local = joblib.load('scaler.pkl')
        df_encoded[numerical_cols] = scaler_local.transform(df_encoded[numerical_cols])
    
    return df_encoded

def prepare_prediction_input(user_inputs):
    """Convert user inputs to properly formatted dataframe"""
    # Create a single-row dataframe with raw values
    df_input = pd.DataFrame([user_inputs])
    
    # Apply the same preprocessing as training data
    df_processed = preprocess_raw_data(df_input)
    
    # Ensure all model columns are present
    for col in columns:
        if col not in df_processed.columns:
            df_processed[col] = 0
    
    # Select and reorder columns to match model
    df_processed = df_processed[columns]
    
    return df_processed

# Sidebar navigation with custom styling
st.sidebar.markdown("<div style='text-align: center; color: white; font-size: 1.5em; font-weight: bold; margin-bottom: 20px;'>📊 Telco Churn Analyzer</div>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# Simple navigation without complex session state
page = st.sidebar.radio(
    "Navigate to:",
    ["🔮 Prediction", "📈 Model Performance", "📊 Data Analysis", "🎯 Feature Analysis"],
    label_visibility="collapsed"
)

# Map display names to page names
page_map = {
    "🔮 Prediction": "Prediction",
    "📈 Model Performance": "Model Performance",
    "📊 Data Analysis": "Data Analysis",
    "🎯 Feature Analysis": "Feature Analysis"
}

page = page_map[page]

# Scroll to top on page load
st.markdown(
    """
    <script>
    window.scrollTo(0, 0);
    </script>
    """,
    unsafe_allow_html=True
)

# Add sidebar footer with info
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style='text-align: center; color: #ddd; font-size: 0.85em; margin-top: 30px;'>
    <p><strong>Telco Churn Predictor</strong></p>
    <p>ML-Powered Customer Retention</p>
    <p style='font-size: 0.75em; margin-top: 10px;'>v1.0</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ==================== PREDICTION PAGE ====================
if page == "Prediction":
    st.markdown("<h1 class='main-header'>🔮 Customer Churn Prediction</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1.5])
    
    with col1:
        st.subheader("📋 Enter Customer Information")
        
        # Collect raw user inputs (same format as raw data)
        user_input = {}
        
        # ===== ACCOUNT INFORMATION =====
        with st.expander("💰 Account Information", expanded=True):
            user_input['Tenure Months'] = st.slider(
                "Tenure (Months)", 0, 72, 24, 
                help="How long the customer has been with us"
            )
            user_input['Monthly Charges'] = st.number_input(
                "Monthly Charges ($)", 0.0, 500.0, 65.0, step=1.0,
                help="Monthly service charges"
            )
            user_input['Total Charges'] = st.number_input(
                "Total Charges ($)", 0.0, 10000.0, 1500.0, step=10.0,
                help="Total accumulated charges"
            )
        
        # ===== PERSONAL INFORMATION =====
        with st.expander("👤 Personal Information", expanded=True):
            col1_per, col2_per = st.columns(2)
            
            with col1_per:
                user_input['Gender'] = st.selectbox(
                    "Gender", ["Female", "Male"],
                    help="Customer gender"
                )
                user_input['Senior Citizen'] = st.selectbox(
                    "Senior Citizen", ["No", "Yes"],
                    help="Is customer a senior citizen?"
                )
            
            with col2_per:
                user_input['Partner'] = st.selectbox(
                    "Has Partner", ["No", "Yes"],
                    help="Does customer have a partner?"
                )
                user_input['Dependents'] = st.selectbox(
                    "Has Dependents", ["No", "Yes"],
                    help="Does customer have dependents?"
                )
        
        # ===== TELEPHONE SERVICES =====
        with st.expander("☎️ Telephone Services"):
            user_input['Phone Service'] = st.selectbox(
                "Phone Service", ["No", "Yes"],
                help="Does customer have phone service?"
            )
            user_input['Multiple Lines'] = st.selectbox(
                "Multiple Lines", ["No", "Yes", "No phone service"],
                help="Does customer have multiple phone lines?"
            )
        
        # ===== INTERNET SERVICES =====
        with st.expander("🌐 Internet Services"):
            user_input['Internet Service Type'] = st.selectbox(
                "Internet Service Type", ["DSL", "Fiber optic", "No"],
                help="Type of internet service"
            )
            
            has_internet = user_input['Internet Service Type'] != "No"
            
            if not has_internet:
                st.warning("⚠️ Without internet service, add-on services are not available and will be locked.")
        
        # ===== ADD-ON SERVICES =====
        with st.expander("🛡️ Add-On Services"):
            if not has_internet:
                st.info("🔒 Internet add-ons are locked because Internet Service Type is 'No'.")
            
            col1_addon, col2_addon = st.columns(2)
            
            with col1_addon:
                user_input['Streaming TV'] = st.selectbox(
                    "Streaming TV", 
                    ["No", "Yes", "No internet service"],
                    disabled=not has_internet,
                    help="Does customer subscribe to streaming TV?" if has_internet else "Locked - requires internet service"
                )
                user_input['Streaming Movies'] = st.selectbox(
                    "Streaming Movies", 
                    ["No", "Yes", "No internet service"],
                    disabled=not has_internet,
                    help="Does customer subscribe to streaming movies?" if has_internet else "Locked - requires internet service"
                )
                user_input['Online Security'] = st.selectbox(
                    "Online Security", 
                    ["No", "Yes", "No internet service"],
                    disabled=not has_internet,
                    help="Does customer have online security?" if has_internet else "Locked - requires internet service"
                )
            
            with col2_addon:
                user_input['Online Backup'] = st.selectbox(
                    "Online Backup", 
                    ["No", "Yes", "No internet service"],
                    disabled=not has_internet,
                    help="Does customer have online backup?" if has_internet else "Locked - requires internet service"
                )
                user_input['Device Protection Plan'] = st.selectbox(
                    "Device Protection Plan", 
                    ["No", "Yes", "No internet service"],
                    disabled=not has_internet,
                    help="Does customer have device protection?" if has_internet else "Locked - requires internet service"
                )
                user_input['Tech Support'] = st.selectbox(
                    "Tech Support", 
                    ["No", "Yes", "No internet service"],
                    disabled=not has_internet,
                    help="Does customer have tech support?" if has_internet else "Locked - requires internet service"
                )
        
        # ===== BILLING INFORMATION =====
        with st.expander("💳 Billing Information", expanded=True):
            col1_bill, col2_bill = st.columns(2)
            
            with col1_bill:
                user_input['Contract'] = st.selectbox(
                    "Contract Type", ["Month-to-month", "One year", "Two year"],
                    help="Customer contract type"
                )
                user_input['Paperless Billing'] = st.selectbox(
                    "Paperless Billing", ["No", "Yes"],
                    help="Does customer use paperless billing?"
                )
            
            with col2_bill:
                user_input['Payment Method'] = st.selectbox(
                    "Payment Method", 
                    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                    help="Customer payment method"
                )
        
        # ===== AUTO-SET LOCKED SERVICES =====
        # When there's no internet service, set add-on services to "No internet service"
        if not has_internet:
            user_input['Streaming TV'] = "No internet service"
            user_input['Streaming Movies'] = "No internet service"
            user_input['Online Security'] = "No internet service"
            user_input['Online Backup'] = "No internet service"
            user_input['Device Protection Plan'] = "No internet service"
            user_input['Tech Support'] = "No internet service"
    
    with col2:
        st.subheader("🎯 Prediction Result")
        
        try:
            # Convert user input to properly formatted dataframe and preprocess
            prediction_df = prepare_prediction_input(user_input)
            
            # Make prediction
            churn_prob = model.predict_proba(prediction_df)[0]
            churn_pred = model.predict(prediction_df)[0]
            
            # Display result
            st.write("")
            st.write("")
            
            if churn_pred == 1:
                st.error("⚠️ HIGH CHURN RISK")
                churn_percentage = churn_prob[1] * 100
            else:
                st.success("✅ LOW CHURN RISK")
                churn_percentage = churn_prob[1] * 100
            
            st.write("")
            st.metric("Churn Probability", f"{churn_percentage:.2f}%")
            st.metric("Retention Probability", f"{(100 - churn_percentage):.2f}%")
            
            st.write("")
            st.write("---")
            
            # Display probability gauge
            fig, ax = plt.subplots(figsize=(8, 4))
            
            colors = ['#2ecc71' if churn_percentage < 50 else '#e74c3c']
            bars = ax.barh(['Churn Risk'], [churn_percentage], color='#e74c3c', height=0.3)
            ax.barh(['Churn Risk'], [100 - churn_percentage], left=[churn_percentage], color='#2ecc71', height=0.3)
            
            ax.set_xlim(0, 100)
            ax.set_xlabel('Probability (%)', fontsize=12)
            ax.set_title('Churn Risk Assessment', fontsize=14, fontweight='bold')
            
            # Add percentage labels
            ax.text(churn_percentage/2, 0, f"{churn_percentage:.1f}%", 
                    ha='center', va='center', fontsize=12, fontweight='bold', color='white')
            ax.text(churn_percentage + (100-churn_percentage)/2, 0, f"{100-churn_percentage:.1f}%", 
                    ha='center', va='center', fontsize=12, fontweight='bold', color='white')
            
            ax.set_yticks([0])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            
            st.pyplot(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("Please check that all input fields are properly filled.")

# ==================== MODEL PERFORMANCE PAGE ====================
elif page == "Model Performance":
    st.markdown("<h1 class='main-header'>📊 Model Performance Metrics</h1>", unsafe_allow_html=True)
    
    try:
        # Prepare data for model evaluation using the same preprocessing
        df_raw = load_data()
        
        st.info(f"📊 Total records loaded: {len(df_raw)}")
        
        # First, handle the Total Charges conversion early
        df_raw['Total Charges'] = pd.to_numeric(df_raw['Total Charges'], errors='coerce')
        
        # Count NaN values before filtering
        nan_count = df_raw[['Total Charges', 'Tenure Months', 'Monthly Charges']].isna().sum().sum()
        st.info(f"Records with missing values: {nan_count}")
        
        # Drop rows with NaN values in numeric columns
        df_clean = df_raw.dropna(subset=['Total Charges', 'Tenure Months', 'Monthly Charges'])
        
        st.info(f"Records after cleaning: {len(df_clean)}")
        
        if len(df_clean) < 1:
            st.error("❌ No valid data available after preprocessing. Dataset may have too many missing values.")
            st.stop()
        
        cols_to_drop = ['Churn Value']
        X_raw = df_clean.drop(columns=cols_to_drop)
        y = df_clean['Churn Value'].reset_index(drop=True)
        
        # Apply the same preprocessing as training
        X_processed = preprocess_raw_data(X_raw)
        
        if X_processed is None:
            st.error("❌ Data preprocessing failed.")
            st.stop()
        
        # Ensure arrays have the same length
        if len(X_processed) != len(y):
            st.warning(f"⚠️ Data length mismatch after processing. X: {len(X_processed)}, y: {len(y)}")
            # Align them
            min_len = min(len(X_processed), len(y))
            X_processed = X_processed.iloc[:min_len]
            y = y.iloc[:min_len]
        
        # Ensure all model columns are present
        for col in columns:
            if col not in X_processed.columns:
                X_processed[col] = 0
        
        X_processed = X_processed[columns]
        
        # Make predictions
        y_pred = model.predict(X_processed)
        y_prob = model.predict_proba(X_processed)[:, 1]
        
        # Calculate metrics
        accuracy = (y_pred == y.values).mean()
        cm = confusion_matrix(y, y_pred)
        roc_auc = roc_auc_score(y, y_prob)
        fpr, tpr, thresholds = roc_curve(y, y_prob)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{accuracy:.4f}")
        with col2:
            st.metric("ROC-AUC Score", f"{roc_auc:.4f}")
        with col3:
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn)
            st.metric("Sensitivity", f"{sensitivity:.4f}")
        with col4:
            specificity = tn / (tn + fp)
            st.metric("Specificity", f"{specificity:.4f}")
        
        st.write("---")
        
        # Confusion Matrix
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                        xticklabels=['No Churn', 'Churn'],
                        yticklabels=['No Churn', 'Churn'])
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            st.pyplot(fig, use_container_width=True)
        
        with col2:
            st.subheader("Classification Report")
            report = classification_report(y, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).round(4)
            st.dataframe(report_df, use_container_width=True)
        
        st.write("---")
        
        # ROC Curve
        st.subheader("ROC Curve")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(fpr, tpr, label=f'Logistic Regression (AUC = {roc_auc:.4f})', linewidth=2)
        ax.plot([0, 1], [0, 1], '--', label='Random Classifier', linewidth=2)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading model performance metrics: {str(e)}")
        st.info("Please ensure the model and scaler files are available.")

# ==================== DATA ANALYSIS PAGE ====================
elif page == "Data Analysis":
    st.markdown("<h1 class='main-header'>📈 Exploratory Data Analysis</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Churn Distribution")
        fig, ax = plt.subplots(figsize=(8, 6))
        churn_counts = df['Churn Value'].value_counts()
        colors = ['#2ecc71', '#e74c3c']
        ax.pie(churn_counts, labels=['No Churn', 'Churn'], autopct='%1.1f%%',
               colors=colors, startangle=90)
        ax.set_title('Customer Churn Distribution', fontweight='bold')
        st.pyplot(fig, use_container_width=True)
    
    with col2:
        st.subheader("Churn Statistics")
        total_customers = len(df)
        churned = (df['Churn Value'] == 1).sum()
        retained = (df['Churn Value'] == 0).sum()
        churn_rate = (churned / total_customers) * 100
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Total Customers", total_customers)
            st.metric("Churned", churned)
        with col_b:
            st.metric("Retained", retained)
            st.metric("Churn Rate", f"{churn_rate:.2f}%")
    
    st.write("---")
    
    # Tenure Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Tenure by Churn Status")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist([df[df['Churn Value'] == 0]['Tenure Months'],
                 df[df['Churn Value'] == 1]['Tenure Months']],
                bins=20, label=['Retained', 'Churned'],
                color=['#2ecc71', '#e74c3c'], alpha=0.7)
        ax.set_xlabel('Tenure (Months)')
        ax.set_ylabel('Number of Customers')
        ax.set_title('Tenure Distribution')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig, use_container_width=True)
    
    with col2:
        st.subheader("Monthly Charges by Churn Status")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist([df[df['Churn Value'] == 0]['Monthly Charges'],
                 df[df['Churn Value'] == 1]['Monthly Charges']],
                bins=20, label=['Retained', 'Churned'],
                color=['#2ecc71', '#e74c3c'], alpha=0.7)
        ax.set_xlabel('Monthly Charges ($)')
        ax.set_ylabel('Number of Customers')
        ax.set_title('Monthly Charges Distribution')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig, use_container_width=True)

# ==================== FEATURE ANALYSIS PAGE ====================
elif page == "Feature Analysis":
    st.markdown("<h1 class='main-header'>🎯 Feature Importance Analysis</h1>", unsafe_allow_html=True)
    
    # Get feature importance from model coefficients
    importance_df = pd.DataFrame({
        'Feature': columns,
        'Coefficient': model.coef_[0]
    }).sort_values(by='Coefficient', key=abs, ascending=True)
    
    # Create visualizations
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Top 15 Features Influencing Churn")
        fig, ax = plt.subplots(figsize=(10, 8))
        
        top_features = importance_df.tail(15)
        colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in top_features['Coefficient']]
        
        ax.barh(range(len(top_features)), top_features['Coefficient'], color=colors)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['Feature'])
        ax.set_xlabel('Coefficient Value')
        ax.set_title('Feature Coefficients (Impact on Churn)', fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(axis='x', alpha=0.3)
        
        st.pyplot(fig, use_container_width=True)
    
    with col2:
        st.subheader("Feature Impact Legend")
        st.info("""
        **Positive Coefficients (Green):** Increase churn likelihood
        
        **Negative Coefficients (Red):** Decrease churn likelihood
        
        **Larger values:** Stronger impact on churn prediction
        """)
    
    st.write("---")
    
    st.subheader("All Features & Coefficients")
    st.dataframe(importance_df.sort_values(by='Coefficient', ascending=False), 
                 use_container_width=True, height=400)

# Footer
st.write("---")
st.markdown("""
    <div style='text-align: center; color: gray; font-size: 12px;'>
    Telco Customer Churn Prediction | Powered by Streamlit | ML Model: Logistic Regression
    </div>
""", unsafe_allow_html=True)
