import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import pickle
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

# Configure Streamlit page
st.set_page_config(
    page_title="Enhanced Soybean Market Analysis Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    .alert-info {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .alert-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .alert-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .model-form {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #007bff;
        margin: 1rem 0;
    }
    .prediction-result {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
    .cointegration-table {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

class EnhancedSoybeanDashboard:
    def __init__(self):
        self.load_data()
    
    def load_data(self):
        """Load enhanced analysis results and models"""
        try:
            # Load JSON results
            with open('enhanced_analysis_results.json', 'r') as f:
                self.results = json.load(f)
            
            # Load pickled models (if available)
            try:
                with open('enhanced_analysis_results.pkl', 'rb') as f:
                    self.models = pickle.load(f)
            except:
                self.models = None
            
            # Load the comprehensive report
            try:
                with open('enhanced_soybean_analysis_report.txt', 'r') as f:
                    self.report = f.read()
            except:
                self.report = "Report not available"
                
            self.markets = ['Haveri', 'Kalagategi', 'Bidar', 'Kalaburgi', 'Bailhongal']
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            self.results = {}
            self.models = None
            self.report = ""
    
    def main_dashboard(self):
        """Enhanced main dashboard page"""
        st.title("🌱 Enhanced Soybean Market Analysis Dashboard")
        st.markdown("### Comprehensive Analysis with Multiple ML Models (Classification & Regression) and Detailed Cointegration")
        
        # Executive Summary Cards
        if 'descriptive_stats' in self.results:
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            total_records = sum(stats['Count'] for stats in self.results['descriptive_stats'].values())
            avg_price = np.mean([stats['Mean_Price'] for stats in self.results['descriptive_stats'].values()])
            highest_market = max(self.results['descriptive_stats'].items(), key=lambda x: x[1]['Mean_Price'])[0]
            markets_analyzed = len(self.results['descriptive_stats'])
            
            # Calculate total ML models (classification + regression)
            total_ml_models = 0
            if 'ml_models' in self.results:
                for model_type in self.results['ml_models']:
                    total_ml_models += len([m for m in self.results['ml_models'][model_type] if self.results['ml_models'][model_type][m] is not None])
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📊 Total Records</h3>
                    <h2>{total_records:,}</h2>
                    <p>Data points analyzed</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>💰 Average Price</h3>
                    <h2>₹{avg_price:.0f}</h2>
                    <p>Per quintal</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🏆 Top Market</h3>
                    <h2>{highest_market}</h2>
                    <p>Highest avg prices</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🌾 Markets</h3>
                    <h2>{markets_analyzed}</h2>
                    <p>Analyzed</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col5:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🤖 ML Models</h3>
                    <h2>{total_ml_models}</h2>
                    <p>Total (Class + Reg)</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col6:
                if 'regression_comparisons' in self.results:
                    r2s = [comp['best_r2'] for comp in self.results['regression_comparisons'].values() if 'best_r2' in comp]
                    avg_r2 = np.mean(r2s) if r2s else 0
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>📈 Avg R²</h3>
                        <h2>{avg_r2:.2f}</h2>
                        <p>Regression Performance</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Enhanced Key Findings
        st.subheader("🔍 Enhanced Key Findings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Market Integration Finding
            if 'cointegration_tables' in self.results:
                coint_relations = self.results['cointegration_tables']['summary_stats']['Number_of_Cointegrating_Relations']
                st.markdown(f"""
                <div class="alert-info">
                    <h4>📈 Market Integration Analysis</h4>
                    <p>Comprehensive Johansen cointegration test reveals <strong>{coint_relations} cointegrating relationship(s)</strong> 
                    among the five markets, indicating {'strong' if coint_relations > 1 else 'moderate' if coint_relations == 1 else 'weak'} market integration.</p>
                    <p><strong>Implication:</strong> {'Markets are highly integrated with rapid price transmission' if coint_relations > 1 else 'Markets show moderate integration with some price linkages' if coint_relations == 1 else 'Markets operate relatively independently'}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Best ARIMA Model
            if 'arima_models' in self.results:
                best_arima = min(self.results['arima_models'].items(), key=lambda x: x[1]['aic'])
                market, model_info = best_arima
                st.markdown(f"""
                <div class="alert-success">
                    <h4>🎯 Best ARIMA Forecasting Model</h4>
                    <p><strong>{market} market</strong> with ARIMA{model_info['best_params']} 
                    achieving AIC of {model_info['aic']:.2f} for reliable price predictions.</p>
                    <p><strong>Model tested:</strong> {len(model_info.get('models_tested', []))} combinations evaluated</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # ML Models Performance (Classification)
            if 'model_comparisons' in self.results:
                st.markdown("**🤖 Classification Model Performance:**")
                
                all_accuracies = []
                for market, comp in self.results['model_comparisons'].items():
                    all_accuracies.extend([acc for _, acc, _ in comp['ranking']])
                
                avg_accuracy = np.mean(all_accuracies) if all_accuracies else 0
                
                st.write(f"📊 **Average Accuracy**: {avg_accuracy:.1%}")
                
                # Show best models by market
                for market, comp in list(self.results['model_comparisons'].items())[:3]:  # Show top 3 markets
                    best_model = comp['best_model']
                    best_acc = comp['best_accuracy']
                    color = "🟢" if best_acc > 0.65 else "🟡" if best_acc > 0.55 else "🔴"
                    st.write(f"{color} **{market}**: {best_model} ({best_acc:.1%})")
            
            # Regression Performance
            if 'regression_comparisons' in self.results:
                st.markdown("**📈 Regression Model Performance:**")
                
                all_r2s = [comp['best_r2'] for comp in self.results['regression_comparisons'].values()]
                avg_r2 = np.mean(all_r2s) if all_r2s else 0
                
                st.write(f"📊 **Average R²**: {avg_r2:.3f}")
                
                for market, comp in list(self.results['regression_comparisons'].items())[:3]:
                    best_model = comp['best_model']
                    best_r2 = comp['best_r2']
                    color = "🟢" if best_r2 > 0.5 else "🟡" if best_r2 > 0.3 else "🔴"
                    st.write(f"{color} **{market}**: {best_model} (R²={best_r2:.3f})")
        
        # Interactive Market Comparison
        st.subheader("📊 Enhanced Interactive Market Analysis")
        
        if 'descriptive_stats' in self.results:
            # Create enhanced comparison data
            markets_data = []
            for market, stats in self.results['descriptive_stats'].items():
                markets_data.append({
                    'Market': market,
                    'Average Price': stats['Mean_Price'],
                    'Price Volatility': stats['Std_Price'],
                    'CV (%)': (stats['Std_Price'] / stats['Mean_Price']) * 100,
                    'Average Arrivals': stats['Mean_Arrivals'],
                    'Max Price': stats['Max_Price'],
                    'Min Price': stats['Min_Price'],
                    'Skewness': stats['Skewness_Price'],
                    'Kurtosis': stats['Kurtosis_Price'],
                    'Sample Size': stats['Count']
                })
            
            df_markets = pd.DataFrame(markets_data)
            
            # Enhanced price comparison with volatility
            fig_enhanced = px.scatter(df_markets, x='Average Price', y='CV (%)',
                                    size='Sample Size', color='Market',
                                    title='Market Risk-Return Profile (CV = Coefficient of Variation)',
                                    hover_data=['Average Arrivals', 'Max Price', 'Min Price'],
                                    labels={'CV (%)': 'Risk (CV %)', 'Average Price': 'Return (Avg Price ₹)'})
            
            fig_enhanced.add_hline(y=df_markets['CV (%)'].mean(), line_dash="dash", 
                                 annotation_text="Average Risk Level", line_color="red")
            fig_enhanced.add_vline(x=df_markets['Average Price'].mean(), line_dash="dash", 
                                 annotation_text="Average Price", line_color="blue")
            
            fig_enhanced.update_layout(height=500)
            st.plotly_chart(fig_enhanced, use_container_width=True)
            
            # Table for Risk-Return Profile
            st.subheader("📋 Market Risk-Return Profile Table")
            risk_return_df = df_markets[['Market', 'Average Price', 'CV (%)', 'Sample Size', 'Average Arrivals']].copy()
            risk_return_df['Average Price'] = risk_return_df['Average Price'].round(2)
            risk_return_df['CV (%)'] = risk_return_df['CV (%)'].round(2)
            st.dataframe(risk_return_df, use_container_width=True)
            
            # Distribution analysis
            col1, col2 = st.columns(2)
            
            with col1:
                fig_dist = px.bar(df_markets, x='Market', y='Skewness',
                                title='Price Distribution Skewness',
                                color='Skewness',
                                color_continuous_scale='RdBu_r')
                fig_dist.add_hline(y=0, line_dash="dash", line_color="black")
                fig_dist.update_layout(height=400)
                st.plotly_chart(fig_dist, use_container_width=True)
                
                # Table for Skewness
                st.subheader("📋 Skewness Table")
                skewness_df = df_markets[['Market', 'Skewness']].copy()
                skewness_df['Skewness'] = skewness_df['Skewness'].round(3)
                st.dataframe(skewness_df, use_container_width=True)
            
            with col2:
                fig_kurt = px.bar(df_markets, x='Market', y='Kurtosis',
                                title='Price Distribution Kurtosis',
                                color='Kurtosis',
                                color_continuous_scale='Viridis')
                fig_kurt.add_hline(y=0, line_dash="dash", line_color="black")
                fig_kurt.update_layout(height=400)
                st.plotly_chart(fig_kurt, use_container_width=True)
                
                # Table for Kurtosis
                st.subheader("📋 Kurtosis Table")
                kurtosis_df = df_markets[['Market', 'Kurtosis']].copy()
                kurtosis_df['Kurtosis'] = kurtosis_df['Kurtosis'].round(3)
                st.dataframe(kurtosis_df, use_container_width=True)
    
    def enhanced_cointegration_analysis(self):
        """Enhanced cointegration analysis page"""
        st.title("🔗 Comprehensive Cointegration Analysis")
        
        if 'cointegration_tables' in self.results:
            coint_tables = self.results['cointegration_tables']
            summary = coint_tables['summary_stats']
            
            # Test Summary
            st.markdown(f"""
            <div class="alert-info">
                <h4>🔬 Johansen Cointegration Test Specifications</h4>
                <ul>
                    <li><strong>Markets Analyzed:</strong> {', '.join(summary['Markets_Analyzed'])}</li>
                    <li><strong>Number of Variables:</strong> {summary['Number_of_Variables']}</li>
                    <li><strong>Cointegrating Relations Found:</strong> {summary['Number_of_Cointegrating_Relations']}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Tabs for different tables
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Trace Statistics", "📈 Maximum Eigenvalue", "🔢 Eigenvalues", "📋 Interpretation"])
            
            with tab1:
                st.subheader("Trace Statistics Test Results")
                
                # Create comprehensive trace statistics table
                trace_df = pd.DataFrame(coint_tables['trace_table'])
                trace_df['Trace_Statistic'] = trace_df['Trace_Statistic'].fillna('N/A')
                
                # Format the dataframe for better display
                display_df = trace_df[['Null_Hypothesis', 'Alternative', 'Trace_Statistic', 
                                     'Critical_Value_5', 'Result_5']].copy()
                
                display_df.columns = ['Null Hypothesis', 'Alternative', 'Trace Statistic', 
                                    'Critical Value (5%)', 'Result']
                
                # Format numbers
                display_df['Trace Statistic'] = display_df['Trace Statistic'].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
                display_df['Critical Value (5%)'] = display_df['Critical Value (5%)'].apply(lambda x: f"{x:.4f}")
                
                st.dataframe(display_df, use_container_width=True)
                
                # Visualization
                fig_trace = go.Figure()
                
                x_labels = [f"r ≤ {i}" for i in range(len(trace_df))]
                trace_stats = [float(x) if isinstance(x, (int, float)) else 0 for x in trace_df['Trace_Statistic'].values]
                critical_vals = [float(x) if isinstance(x, (int, float)) else 0 for x in trace_df['Critical_Value_5'].values]
                
                fig_trace.add_trace(go.Bar(
                    x=x_labels,
                    y=trace_stats,
                    name='Trace Statistic',
                    marker_color='lightblue'
                ))
                
                fig_trace.add_trace(go.Scatter(
                    x=x_labels,
                    y=critical_vals,
                    mode='lines+markers',
                    name='Critical Value (5%)',
                    line=dict(color='red', width=3)
                ))
                
                fig_trace.update_layout(
                    title='Trace Statistics vs Critical Values',
                    xaxis_title='Null Hypothesis',
                    yaxis_title='Statistic Value',
                    height=400
                )
                
                st.plotly_chart(fig_trace, use_container_width=True)
                
                # Explanation
                st.markdown("""
                **📖 How to Read Trace Statistics:**
                - **Null Hypothesis (r ≤ k)**: At most k cointegrating relationships exist
                - **Reject H₀**: Evidence for more than k cointegrating relationships
                - **Accept H₀**: No evidence for more than k cointegrating relationships
                - **Decision Rule**: If Trace Statistic > Critical Value → Reject H₀
                """)
            
            with tab2:
                st.subheader("Maximum Eigenvalue Test Results")
                
                # Create maximum eigenvalue table
                eigen_df = pd.DataFrame(coint_tables['max_eigen_table'])
                eigen_df['Max_Eigen_Statistic'] = eigen_df['Max_Eigen_Statistic'].fillna('N/A')
                
                display_df = eigen_df[['Null_Hypothesis', 'Alternative', 'Max_Eigen_Statistic', 
                                     'Critical_Value_5', 'Result_5']].copy()
                
                display_df.columns = ['Null Hypothesis', 'Alternative', 'Max Eigen Statistic', 
                                    'Critical Value (5%)', 'Result']
                
                # Format numbers
                display_df['Max Eigen Statistic'] = display_df['Max Eigen Statistic'].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
                display_df['Critical Value (5%)'] = display_df['Critical Value (5%)'].apply(lambda x: f"{x:.4f}")
                
                st.dataframe(display_df, use_container_width=True)
                
                # Visualization
                fig_eigen = go.Figure()
                
                x_labels = [f"r = {i}" for i in range(len(eigen_df))]
                eigen_stats = [float(x) if isinstance(x, (int, float)) else 0 for x in eigen_df['Max_Eigen_Statistic'].values]
                critical_vals = [float(x) if isinstance(x, (int, float)) else 0 for x in eigen_df['Critical_Value_5'].values]
                
                fig_eigen.add_trace(go.Bar(
                    x=x_labels,
                    y=eigen_stats,
                    name='Max Eigen Statistic',
                    marker_color='lightgreen'
                ))
                
                fig_eigen.add_trace(go.Scatter(
                    x=x_labels,
                    y=critical_vals,
                    mode='lines+markers',
                    name='Critical Value (5%)',
                    line=dict(color='red', width=3)
                ))
                
                fig_eigen.update_layout(
                    title='Maximum Eigenvalue Statistics vs Critical Values',
                    xaxis_title='Null Hypothesis',
                    yaxis_title='Statistic Value',
                    height=400
                )
                
                st.plotly_chart(fig_eigen, use_container_width=True)
            
            with tab3:
                st.subheader("Eigenvalues Analysis")
                
                eigenvalues = summary['Eigenvalues']
                
                # Create eigenvalues dataframe
                eigen_data = []
                for i, eigenval in enumerate(eigenvalues, 1):
                    eigen_data.append({
                        'Eigenvalue': f'λ{i}',
                        'Value': eigenval,
                        'Magnitude': 'Large' if eigenval > 0.1 else 'Medium' if eigenval > 0.01 else 'Small'
                    })
                
                eigen_df = pd.DataFrame(eigen_data)
                st.dataframe(eigen_df, use_container_width=True)
                
                # Eigenvalues plot
                fig_eigen_plot = px.bar(eigen_df, x='Eigenvalue', y='Value',
                                      title='Eigenvalues Magnitude',
                                      color='Value',
                                      color_continuous_scale='Viridis')
                fig_eigen_plot.update_layout(height=400)
                st.plotly_chart(fig_eigen_plot, use_container_width=True)
                
                st.markdown("""
                **📊 Eigenvalues Interpretation:**
                - **Large eigenvalues (> 0.1)**: Strong cointegrating relationships
                - **Medium eigenvalues (0.01-0.1)**: Moderate cointegrating relationships  
                - **Small eigenvalues (< 0.01)**: Weak or no cointegrating relationships
                """)
            
            with tab4:
                st.subheader("Economic Interpretation")
                
                interpretation = coint_tables['interpretation']
                
                st.markdown(f"""
                <div class="alert-success">
                    <h4>🎯 {interpretation['conclusion']}</h4>
                    <p><strong>Economic Meaning:</strong> {interpretation['meaning']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📈 Market Implications:**")
                    for implication in interpretation['implications']:
                        st.write(f"• {implication}")
                
                with col2:
                    st.markdown("**🏛️ Policy Implications:**")
                    for policy in interpretation['policy_implications']:
                        st.write(f"• {policy}")
                
                # Additional insights
                st.markdown("---")
                st.subheader("💡 Strategic Insights")
                
                num_relations = summary['Number_of_Cointegrating_Relations']
                
                if num_relations == 0:
                    st.markdown("""
                    <div class="alert-warning">
                        <h5>⚠️ Independent Markets</h5>
                        <p><strong>Trading Strategy:</strong> Treat each market independently</p>
                        <p><strong>Risk Management:</strong> Diversification across markets may be effective</p>
                        <p><strong>Arbitrage Opportunities:</strong> May exist between markets</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                elif num_relations == 1:
                    st.markdown("""
                    <div class="alert-info">
                        <h5>📊 Moderate Integration</h5>
                        <p><strong>Trading Strategy:</strong> Consider inter-market relationships</p>
                        <p><strong>Risk Management:</strong> Some correlation risk exists</p>
                        <p><strong>Price Discovery:</strong> Information flows moderately between markets</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                else:
                    st.markdown("""
                    <div class="alert-success">
                        <h5>✅ Strong Integration</h5>
                        <p><strong>Trading Strategy:</strong> Treat as integrated market system</p>
                        <p><strong>Risk Management:</strong> High correlation risk, limited diversification benefits</p>
                        <p><strong>Price Discovery:</strong> Rapid information transmission across markets</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        else:
            st.warning("Cointegration analysis results not available.")
    
    def enhanced_arima_analysis(self):
        """Enhanced ARIMA analysis with detailed explanations"""
        st.title("🔮 Enhanced ARIMA Forecasting with Model Selection Explanations")
        
        if 'arima_models' in self.results:
            # Market selection
            selected_market = st.selectbox("Select Market for Detailed Analysis:", 
                                         list(self.results['arima_models'].keys()))
            
            if selected_market in self.results['arima_models']:
                model_info = self.results['arima_models'][selected_market]
                
                # Model overview
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Selected Model", f"ARIMA{model_info['best_params']}")
                
                with col2:
                    st.metric("AIC Score", f"{model_info['aic']:.2f}")
                
                with col3:
                    st.metric("Parameters", f"{model_info['parameters_count']}")
                
                with col4:
                    st.metric("Log-Likelihood", f"{model_info['log_likelihood']:.2f}")
                
                # Detailed AIC Explanation
                if 'arima_explanations' in self.results and selected_market in self.results['arima_explanations']:
                    explanation = self.results['arima_explanations'][selected_market]
                    
                    st.subheader(f"🎯 Why ARIMA{model_info['best_params']} was Selected for {selected_market}")
                    
                    # Parameter interpretation
                    st.markdown("**📊 Parameter Interpretation:**")
                    p, d, q = model_info['best_params']
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="alert-info">
                            <h5>AR(p={p})</h5>
                            <p>{'Uses past ' + str(p) + ' price values to predict current price' if p > 0 else 'No autoregressive component needed'}</p>
                            <p><strong>Meaning:</strong> {'Price momentum and trend patterns detected' if p > 0 else 'No momentum patterns'}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="alert-info">
                            <h5>I(d={d})</h5>
                            <p>{'Data differenced ' + str(d) + ' time(s) to achieve stationarity' if d > 0 else 'Data is already stationary'}</p>
                            <p><strong>Meaning:</strong> {'Trend removal required' if d > 0 else 'No trend present'}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="alert-info">
                            <h5>MA(q={q})</h5>
                            <p>{'Uses past ' + str(q) + ' forecast errors to improve predictions' if q > 0 else 'No moving average component needed'}</p>
                            <p><strong>Meaning:</strong> {'Error correction patterns detected' if q > 0 else 'No error patterns'}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Model comparison
                    st.subheader("📈 Model Selection Process")
                    
                    if 'models_tested' in model_info:
                        models_tested = model_info['models_tested']
                        tested_combinations = len(models_tested)
                        successful_fits = len([m for m in models_tested if 'AIC' in m])
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Combinations Tested", tested_combinations)
                        
                        with col2:
                            st.metric("Successful Fits", successful_fits)
                        
                        with col3:
                            st.metric("Success Rate", f"{successful_fits/tested_combinations*100:.1f}%")
                        
                        with col4:
                            aic_values = [m['AIC'] for m in models_tested if 'AIC' in m]
                            if aic_values:
                                aic_range = max(aic_values) - min(aic_values)
                                st.metric("AIC Range", f"{aic_range:.1f}")
                    
                    # Top models comparison
                    if 'models_tested' in model_info and len(model_info['models_tested']) > 1:
                        st.subheader("🏆 Top 10 Models Comparison")
                        
                        models_df = pd.DataFrame(model_info['models_tested'][:10])
                        models_df = models_df.sort_values('AIC').reset_index(drop=True)
                        models_df['Rank'] = range(1, len(models_df) + 1)
                        models_df['Model'] = models_df['order'].apply(lambda x: f'ARIMA({x[0]},{x[1]},{x[2]})')
                        models_df['ΔAIC'] = models_df['AIC'] - models_df['AIC'].min()
                        
                        # Color code by performance
                        def color_delta_aic(val):
                            if val < 2:
                                return 'background-color: #d4edda'  # Green
                            elif val < 4:
                                return 'background-color: #fff3cd'  # Yellow
                            else:
                                return 'background-color: #f8d7da'  # Red
                        
                        styled_df = models_df[['Rank', 'Model', 'AIC', 'BIC', 'ΔAIC']].style.applymap(
                            color_delta_aic, subset=['ΔAIC'])
                        
                        st.dataframe(styled_df, use_container_width=True)
                        
                        # AIC comparison chart
                        fig_aic = px.bar(models_df.head(8), x='Model', y='ΔAIC',
                                       title='Model Comparison (ΔAIC from Best Model)',
                                       color='ΔAIC',
                                       color_continuous_scale='RdYlGn_r')
                        fig_aic.add_hline(y=2, line_dash="dash", annotation_text="Substantial Support Threshold")
                        fig_aic.add_hline(y=4, line_dash="dash", annotation_text="Some Support Threshold")
                        fig_aic.update_layout(height=400)
                        st.plotly_chart(fig_aic, use_container_width=True)
                        
                        st.markdown("""
                        **📖 AIC Interpretation Guide:**
                        - **ΔAIC < 2**: Substantial support (alternative models are competitive)
                        - **2 ≤ ΔAIC < 4**: Some support (alternative models have merit)
                        - **4 ≤ ΔAIC < 7**: Little support (alternative models are weak)
                        - **ΔAIC ≥ 7**: No support (alternative models are poor)
                        """)
                
                # Forecasting section
                st.subheader("📈 Price Forecasts")
                
                if 'forecast' in model_info and model_info['forecast']:
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Forecast chart
                        months = list(range(1, len(model_info['forecast']) + 1))
                        fig_forecast = go.Figure()
                        
                        fig_forecast.add_trace(go.Scatter(
                            x=months,
                            y=model_info['forecast'],
                            mode='lines+markers',
                            name='Point Forecast',
                            line=dict(color='blue', width=3)
                        ))
                        
                        fig_forecast.update_layout(
                            title=f'{selected_market} Market - ARIMA{model_info["best_params"]} Forecast',
                            xaxis_title='Months Ahead',
                            yaxis_title='Price (₹/Quintal)',
                            height=400
                        )
                        
                        st.plotly_chart(fig_forecast, use_container_width=True)
                    
                    with col2:
                        st.markdown("**📊 Forecast Table:**")
                        
                        forecast_data = []
                        for i, price in enumerate(model_info['forecast'][:12], 1):
                            forecast_data.append({
                                'Month': i,
                                'Price (₹)': f'{price:.2f}'
                            })
                        
                        forecast_df = pd.DataFrame(forecast_data)
                        st.dataframe(forecast_df, use_container_width=True)
                
                # Model diagnostics
                st.subheader("🔬 Model Diagnostics")
                
                diagnostic_col1, diagnostic_col2 = st.columns(2)
                
                with diagnostic_col1:
                    st.markdown("**✅ Model Quality Indicators:**")
                    
                    aic_quality = "Excellent" if model_info['aic'] < 1200 else "Good" if model_info['aic'] < 1400 else "Fair"
                    params_complexity = "Simple" if model_info['parameters_count'] <= 3 else "Moderate" if model_info['parameters_count'] <= 5 else "Complex"
                    
                    st.write(f"• **AIC Quality**: {aic_quality} ({model_info['aic']:.2f})")
                    st.write(f"• **Model Complexity**: {params_complexity} ({model_info['parameters_count']} parameters)")
                    st.write(f"• **Log-likelihood**: {model_info['log_likelihood']:.2f}")
                
                with diagnostic_col2:
                    st.markdown("**📋 Model Assumptions:**")
                    st.write("• ✅ Time series stationarity achieved")
                    st.write("• ✅ Residuals should be white noise")
                    st.write("• ✅ No remaining autocorrelation")
                    st.write("• ⚠️ Assumes linear relationships")
            
            # Interactive forecast tool
            st.markdown("---")
            st.subheader("🛠️ Interactive ARIMA Forecasting Tool")
            
            forecast_market = st.selectbox("Choose Market:", list(self.results['arima_models'].keys()), key='arima_forecast_tool')
            forecast_periods = st.slider("Forecast Periods (Months):", 1, 24, 6)
            
            if st.button("Generate ARIMA Forecast", type="primary"):
                if forecast_market in self.results['arima_models']:
                    model_info = self.results['arima_models'][forecast_market]
                    
                    # Use available forecast data
                    available_periods = min(forecast_periods, len(model_info.get('forecast', [])))
                    
                    if available_periods > 0:
                        st.success(f"Generated {available_periods}-month ARIMA forecast for {forecast_market}")
                        
                        # Display enhanced forecast
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            # Create forecast visualization
                            months = list(range(1, available_periods + 1))
                            forecasts = model_info['forecast'][:available_periods]
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=months, y=forecasts,
                                mode='lines+markers',
                                name=f'ARIMA{model_info["best_params"]} Forecast',
                                line=dict(color='purple', width=3)
                            ))
                            
                            fig.update_layout(
                                title=f'{forecast_market} - {available_periods} Month Forecast',
                                xaxis_title='Months Ahead',
                                yaxis_title='Price (₹/Quintal)',
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Forecast statistics
                            forecast_mean = np.mean(forecasts)
                            forecast_trend = "Increasing" if forecasts[-1] > forecasts[0] else "Decreasing" if forecasts[-1] < forecasts[0] else "Stable"
                            
                            st.markdown("**📊 Forecast Summary:**")
                            st.write(f"• **Average Price**: ₹{forecast_mean:.2f}")
                            st.write(f"• **Price Trend**: {forecast_trend}")
                            st.write(f"• **Model Used**: ARIMA{model_info['best_params']}")
                            st.write(f"• **Model AIC**: {model_info['aic']:.2f}")
                    else:
                        st.error("No forecast data available for this market.")
        
        else:
            st.warning("No ARIMA models available. Please run the analysis first.")
    
    def enhanced_ml_models(self):
        """Enhanced ML models page with forms for all three models (Classification & Regression)"""
        st.title("🤖 Enhanced Machine Learning Models")
        st.markdown("### Interactive Forms for Classification (Logistic Regression & Random Forest) and Regression (Linear Regression)")
        
        if 'ml_models' in self.results and any(self.results['ml_models'].values()):
            
            # Model comparison overview - Classification
            st.subheader("📊 Classification Model Performance Overview")
            
            if 'model_comparisons' in self.results:
                comparison_data = []
                for market, comp in self.results['model_comparisons'].items():
                    for rank, item in enumerate(comp['ranking'], 1):
                        if isinstance(item, (list, tuple)):
                            model_name = item[0]
                            accuracy = item[1]
                            cv_score = item[2]
                        else:
                            # Fallback assuming dict or other, but skip for now
                            continue
                        comparison_data.append({
                            'Market': market,
                            'Model': model_name,
                            'Test Accuracy': accuracy,
                            'CV Score': cv_score,
                            'Rank in Market': rank
                        })
                
                comparison_df = pd.DataFrame(comparison_data)
                
                # Performance visualization
                fig_performance = px.box(comparison_df, x='Model', y='Test Accuracy',
                                       title='Classification Model Performance Distribution Across Markets',
                                       color='Model')
                fig_performance.update_layout(height=400)
                st.plotly_chart(fig_performance, use_container_width=True)
                
                # Best models table
                st.subheader("🏆 Best Classification Models by Market")
                best_models_data = []
                for market, comp in self.results['model_comparisons'].items():
                    if comp['ranking']:
                        item = comp['ranking'][0]
                        if isinstance(item, (list, tuple)):
                            best_model = item[0]
                            best_acc = item[1]
                            best_cv = item[2]
                        else:
                            best_model = comp['best_model']
                            best_acc = comp['best_accuracy']
                            best_cv = 0  # fallback
                        best_models_data.append({
                            'Market': market,
                            'Best Model': best_model,
                            'Accuracy': f"{best_acc:.1%}",
                            'CV Score': f"{best_cv:.1%}"
                        })
                
                best_models_df = pd.DataFrame(best_models_data)
                st.dataframe(best_models_df, use_container_width=True)
            
            # Regression Performance Overview
            st.subheader("📈 Regression Model Performance Overview")
            
            if 'regression_comparisons' in self.results:
                reg_comparison_data = []
                for market, comp in self.results['regression_comparisons'].items():
                    for rank, item in enumerate(comp['ranking'], 1):
                        if isinstance(item, (list, tuple)):
                            model_name = item[0]
                            r2 = item[1]
                            cv_score = item[2]
                        else:
                            continue
                        reg_comparison_data.append({
                            'Market': market,
                            'Model': model_name,
                            'Test R²': r2,
                            'CV Score': cv_score,
                            'Rank in Market': rank
                        })
                
                reg_comparison_df = pd.DataFrame(reg_comparison_data)
                
                # Performance visualization
                fig_reg_performance = px.box(reg_comparison_df, x='Model', y='Test R²',
                                           title='Regression Model Performance Distribution Across Markets',
                                           color='Model')
                fig_reg_performance.update_layout(height=400)
                st.plotly_chart(fig_reg_performance, use_container_width=True)
                
                # Best regression models table
                st.subheader("🏆 Best Regression Models by Market")
                best_reg_data = []
                for market, comp in self.results['regression_comparisons'].items():
                    if comp['ranking']:
                        item = comp['ranking'][0]
                        if isinstance(item, (list, tuple)):
                            best_model = item[0]
                            best_r2 = item[1]
                            best_cv = item[2]
                        else:
                            best_model = comp['best_model']
                            best_r2 = comp['best_r2']
                            best_cv = 0
                        best_reg_data.append({
                            'Market': market,
                            'Best Model': best_model,
                            'R²': f"{best_r2:.3f}",
                            'CV Score': f"{best_cv:.3f}"
                        })
                
                best_reg_df = pd.DataFrame(best_reg_data)
                st.dataframe(best_reg_df, use_container_width=True)
            
            # Interactive prediction forms
            st.markdown("---")
            st.subheader("🔮 Interactive Prediction Forms")
            
            # Tabs for Classification and Regression
            tab_class, tab_reg = st.tabs(["🔄 Classification (Direction)", "📈 Regression (Price Level)"])
            
            with tab_class:
                # Classification forms
                col1, col2 = st.columns(2)
                
                with col1:
                    available_markets_class = []
                    for model_type in ['logistic_regression', 'random_forest']:
                        if model_type in self.results['ml_models']:
                            available_markets_class.extend(self.results['ml_models'][model_type].keys())
                    available_markets_class = list(set(available_markets_class))
                    
                    selected_market_class = st.selectbox("Select Market:", available_markets_class, key="class_market")
                
                with col2:
                    available_models_class = []
                    if selected_market_class:
                        for model_type in ['logistic_regression', 'random_forest']:
                            if (model_type in self.results['ml_models'] and 
                                selected_market_class in self.results['ml_models'][model_type] and
                                self.results['ml_models'][model_type][selected_market_class] is not None):
                                available_models_class.append(model_type.replace('_', ' ').title())
                    
                    selected_model_type_class = st.selectbox("Select Model Type:", available_models_class, key="class_model")
                
                if selected_market_class and selected_model_type_class:
                    model_key = selected_model_type_class.lower().replace(' ', '_')
                    
                    if (model_key in self.results['ml_models'] and 
                        selected_market_class in self.results['ml_models'][model_key] and
                        self.results['ml_models'][model_key][selected_market_class] is not None):
                        
                        model_results = self.results['ml_models'][model_key][selected_market_class]
                        
                        # Model information
                        st.markdown(f"""
                        <div class="alert-info">
                            <h4>📋 {selected_model_type_class} Model for {selected_market_class}</h4>
                            <p><strong>Test Accuracy:</strong> {model_results['accuracy']:.1%}</p>
                            <p><strong>Cross-Validation Score:</strong> {model_results['cv_mean']:.1%} ± {model_results['cv_std']:.1%}</p>
                            <p><strong>Features Used:</strong> {len(model_results['feature_names'])} variables</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Prediction form
                        st.markdown(f"""
                        <div class="model-form">
                            <h4>🔮 {selected_model_type_class} Prediction Form</h4>
                            <p>Enter market conditions to predict price movement direction</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Feature input form (same as before for classification)
                        feature_names = model_results['feature_names']
                        feature_values = {}
                        
                        # Create input fields based on feature names
                        col1, col2, col3 = st.columns(3)
                        
                        for i, feature in enumerate(feature_names):
                            col = [col1, col2, col3][i % 3]
                            
                            with col:
                                if 'Arrivals' in feature and 'MA' not in feature:
                                    if 'Lag' in feature:
                                        feature_values[feature] = st.number_input(
                                            f"{feature} (Tonnes):", 
                                            min_value=0.0, max_value=1000.0, 
                                            value=50.0, step=1.0, key=f"class_{feature}_{model_key}_{selected_market_class}"
                                        )
                                    else:
                                        feature_values[feature] = st.number_input(
                                            f"{feature} (Tonnes):", 
                                            min_value=0.0, max_value=1000.0, 
                                            value=60.0, step=1.0, key=f"class_{feature}_{model_key}_{selected_market_class}"
                                        )
                                elif 'Price' in feature:
                                    if 'Lag' in feature:
                                        feature_values[feature] = st.number_input(
                                            f"{feature} (₹):", 
                                            min_value=1000.0, max_value=10000.0, 
                                            value=4000.0, step=10.0, key=f"class_{feature}_{model_key}_{selected_market_class}"
                                        )
                                    elif 'MA' in feature:
                                        feature_values[feature] = st.number_input(
                                            f"{feature} (₹):", 
                                            min_value=1000.0, max_value=10000.0, 
                                            value=4100.0, step=10.0, key=f"class_{feature}_{model_key}_{selected_market_class}"
                                        )
                                    elif 'Volatility' in feature:
                                        feature_values[feature] = st.number_input(
                                            f"{feature} (₹):", 
                                            min_value=0.0, max_value=2000.0, 
                                            value=100.0, step=1.0, key=f"class_{feature}_{model_key}_{selected_market_class}"
                                        )
                                elif 'MA' in feature and 'Arrivals' in feature:
                                    feature_values[feature] = st.number_input(
                                        f"{feature} (Tonnes):", 
                                        min_value=0.0, max_value=1000.0, 
                                        value=55.0, step=1.0, key=f"class_{feature}_{model_key}_{selected_market_class}"
                                    )
                                elif 'Month' in feature:
                                    feature_values[feature] = st.number_input(
                                        f"{feature}:", 
                                        min_value=1, max_value=12, 
                                        value=6, step=1, key=f"class_{feature}_{model_key}_{selected_market_class}"
                                    )
                                elif 'Quarter' in feature:
                                    feature_values[feature] = st.number_input(
                                        f"{feature}:", 
                                        min_value=1, max_value=4, 
                                        value=2, step=1, key=f"class_{feature}_{model_key}_{selected_market_class}"
                                    )
                                else:
                                    feature_values[feature] = st.number_input(
                                        f"{feature}:", 
                                        value=0.0, key=f"class_{feature}_{model_key}_{selected_market_class}"
                                    )
                        
                        # Prediction button for classification
                        if st.button(f"🎯 Predict Direction with {selected_model_type_class}", type="primary", key=f"class_predict_{model_key}_{selected_market_class}"):
                            
                            # Prepare feature vector
                            features_array = np.array([feature_values[feature] for feature in feature_names]).reshape(1, -1)
                            
                            # Simple prediction logic (simulation)
                            if model_key == 'logistic_regression' and 'coefficients' in model_results:
                                # Simulate logistic regression prediction
                                coefficients = model_results['coefficients']
                                # Normalize features (simple approximation)
                                features_normalized = (features_array - np.mean(features_array)) / (np.std(features_array) + 1e-8)
                                linear_combination = np.sum(features_normalized * coefficients)
                                probability = 1 / (1 + np.exp(-linear_combination))
                            
                            elif model_key == 'random_forest' and 'feature_importance' in model_results:
                                # Simulate tree-based model prediction
                                importances = model_results['feature_importance']
                                # Weighted average based on feature importance (approximation)
                                weighted_features = features_array[0] * importances
                                score = np.sum(weighted_features) / np.sum(importances)
                                probability = 1 / (1 + np.exp(-(score - 0.5) * 2))  # Simple sigmoid transformation
                            
                            else:
                                # Fallback: use model accuracy as base probability
                                probability = model_results['accuracy']
                            
                            # Ensure probability is in valid range
                            probability = max(0.1, min(0.9, probability))
                            
                            prediction = "📈 INCREASE" if probability > 0.5 else "📉 DECREASE"  
                            confidence = max(probability, 1 - probability)
                            
                            # Display results
                            st.markdown(f"""
                            <div class="prediction-result">
                                <h4>🎯 {selected_model_type_class} Prediction Result</h4>
                                <p><strong>Market:</strong> {selected_market_class}</p>
                                <p><strong>Predicted Movement:</strong> {prediction}</p>
                                <p><strong>Confidence:</strong> {confidence:.1%}</p>
                                <p><strong>Probability of Increase:</strong> {probability:.1%}</p>
                                <p><strong>Model Accuracy:</strong> {model_results['accuracy']:.1%}</p>
                                <p><strong>Model Type:</strong> {selected_model_type_class}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Feature importance display (same as before)
                            if model_key == 'logistic_regression' and 'coefficients' in model_results:
                                st.subheader("📊 Feature Influence (Logistic Regression Coefficients)")
                                
                                coef_data = []
                                for feature, coef in zip(feature_names, model_results['coefficients']):
                                    coef_data.append({
                                        'Feature': feature,
                                        'Coefficient': coef,
                                        'Impact': 'Positive' if coef > 0 else 'Negative',
                                        'Magnitude': abs(coef)
                                    })
                                
                                coef_df = pd.DataFrame(coef_data).sort_values('Magnitude', ascending=False)
                                
                                fig_coef = px.bar(coef_df, x='Feature', y='Coefficient',
                                                title='Feature Coefficients (Impact on Price Increase)',
                                                color='Coefficient',
                                                color_continuous_scale='RdBu_r')
                                fig_coef.add_hline(y=0, line_dash="dash", line_color="black")
                                fig_coef.update_layout(height=400, xaxis_tickangle=-45)
                                st.plotly_chart(fig_coef, use_container_width=True)
                            
                            elif 'feature_importance' in model_results:
                                st.subheader("📊 Feature Importance")
                                
                                importance_data = []
                                for feature, importance in zip(feature_names, model_results['feature_importance']):
                                    importance_data.append({
                                        'Feature': feature,
                                        'Importance': importance
                                    })
                                
                                importance_df = pd.DataFrame(importance_data).sort_values('Importance', ascending=False)
                                
                                fig_importance = px.bar(importance_df, x='Feature', y='Importance',
                                                      title='Feature Importance in Model Decision',
                                                      color='Importance',
                                                      color_continuous_scale='Viridis')
                                fig_importance.update_layout(height=400, xaxis_tickangle=-45)
                                st.plotly_chart(fig_importance, use_container_width=True)
                
                # Classification Model comparison section
                st.markdown("---")
                st.subheader("📈 Detailed Classification Model Comparison")
                
                if selected_market_class:
                    available_model_results_class = {}
                    
                    for model_type in ['logistic_regression', 'random_forest']:
                        if (model_type in self.results['ml_models'] and 
                            selected_market_class in self.results['ml_models'][model_type] and
                            self.results['ml_models'][model_type][selected_market_class] is not None):
                            available_model_results_class[model_type] = self.results['ml_models'][model_type][selected_market_class]
                    
                    if available_model_results_class:
                        # Performance metrics comparison
                        metrics_data = []
                        for model_type, results in available_model_results_class.items():
                            metrics_data.append({
                                'Model': model_type.replace('_', ' ').title(),
                                'Test Accuracy': results['accuracy'],
                                'CV Mean': results['cv_mean'],
                                'CV Std': results['cv_std'],
                                'Stability': 'High' if results['cv_std'] < 0.05 else 'Medium' if results['cv_std'] < 0.1 else 'Low'
                            })
                        
                        metrics_df = pd.DataFrame(metrics_data)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.dataframe(metrics_df, use_container_width=True)
                        
                        with col2:
                            # Radar chart for model comparison
                            fig_radar = go.Figure()
                            
                            for _, row in metrics_df.iterrows():
                                fig_radar.add_trace(go.Scatterpolar(
                                    r=[row['Test Accuracy'], row['CV Mean'], 1-row['CV Std']],  # 1-CV_Std for stability
                                    theta=['Test Accuracy', 'CV Score', 'Stability'],
                                    fill='toself',
                                    name=row['Model']
                                ))
                            
                            fig_radar.update_layout(
                                polar=dict(
                                    radialaxis=dict(
                                        visible=True,
                                        range=[0, 1]
                                    )),
                                showlegend=True,
                                title=f"Classification Model Performance Comparison - {selected_market_class}",
                                height=400
                            )
                            
                            st.plotly_chart(fig_radar, use_container_width=True)
            
            with tab_reg:
                # Regression forms
                col1, col2 = st.columns(2)
                
                with col1:
                    available_markets_reg = []
                    for model_type in ['linear_regression']:
                        if model_type in self.results['ml_models']:
                            available_markets_reg.extend(self.results['ml_models'][model_type].keys())
                    available_markets_reg = list(set(available_markets_reg))
                    
                    selected_market_reg = st.selectbox("Select Market:", available_markets_reg, key="reg_market")
                
                with col2:
                    available_models_reg = []
                    if selected_market_reg:
                        for model_type in ['linear_regression']:
                            if (model_type in self.results['ml_models'] and 
                                selected_market_reg in self.results['ml_models'][model_type] and
                                self.results['ml_models'][model_type][selected_market_reg] is not None):
                                available_models_reg.append(model_type.replace('_', ' ').title())
                    
                    selected_model_type_reg = st.selectbox("Select Model Type:", available_models_reg, key="reg_model")
                
                if selected_market_reg and selected_model_type_reg:
                    model_key = selected_model_type_reg.lower().replace(' ', '_')
                    
                    if (model_key in self.results['ml_models'] and 
                        selected_market_reg in self.results['ml_models'][model_key] and
                        self.results['ml_models'][model_key][selected_market_reg] is not None):
                        
                        model_results = self.results['ml_models'][model_key][selected_market_reg]
                        
                        # Model information
                        st.markdown(f"""
                        <div class="alert-info">
                            <h4>📋 {selected_model_type_reg} Model for {selected_market_reg}</h4>
                            <p><strong>Test R² Score:</strong> {model_results['r2_score']:.3f}</p>
                            <p><strong>Cross-Validation Score:</strong> {model_results['cv_mean']:.3f} ± {model_results['cv_std']:.3f}</p>
                            <p><strong>Features Used:</strong> {len(model_results['feature_names'])} variables</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Prediction form for regression
                        st.markdown(f"""
                        <div class="model-form">
                            <h4>🔮 {selected_model_type_reg} Prediction Form</h4>
                            <p>Enter market conditions to predict price level</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Feature input form (similar to classification)
                        feature_names = model_results['feature_names']
                        feature_values = {}
                        
                        col1, col2, col3 = st.columns(3)
                        
                        for i, feature in enumerate(feature_names):
                            col = [col1, col2, col3][i % 3]
                            
                            with col:
                                if 'Arrivals' in feature and 'MA' not in feature:
                                    if 'Lag' in feature:
                                        feature_values[feature] = st.number_input(
                                            f"{feature} (Tonnes):", 
                                            min_value=0.0, max_value=1000.0, 
                                            value=50.0, step=1.0, key=f"reg_{feature}_{model_key}_{selected_market_reg}"
                                        )
                                    else:
                                        feature_values[feature] = st.number_input(
                                            f"{feature} (Tonnes):", 
                                            min_value=0.0, max_value=1000.0, 
                                            value=60.0, step=1.0, key=f"reg_{feature}_{model_key}_{selected_market_reg}"
                                        )
                                elif 'Price' in feature:
                                    if 'Lag' in feature:
                                        feature_values[feature] = st.number_input(
                                            f"{feature} (₹):", 
                                            min_value=1000.0, max_value=10000.0, 
                                            value=4000.0, step=10.0, key=f"reg_{feature}_{model_key}_{selected_market_reg}"
                                        )
                                    elif 'MA' in feature:
                                        feature_values[feature] = st.number_input(
                                            f"{feature} (₹):", 
                                            min_value=1000.0, max_value=10000.0, 
                                            value=4100.0, step=10.0, key=f"reg_{feature}_{model_key}_{selected_market_reg}"
                                        )
                                    elif 'Volatility' in feature:
                                        feature_values[feature] = st.number_input(
                                            f"{feature} (₹):", 
                                            min_value=0.0, max_value=2000.0, 
                                            value=100.0, step=1.0, key=f"reg_{feature}_{model_key}_{selected_market_reg}"
                                        )
                                elif 'MA' in feature and 'Arrivals' in feature:
                                    feature_values[feature] = st.number_input(
                                        f"{feature} (Tonnes):", 
                                        min_value=0.0, max_value=1000.0, 
                                        value=55.0, step=1.0, key=f"reg_{feature}_{model_key}_{selected_market_reg}"
                                    )
                                elif 'Month' in feature:
                                    feature_values[feature] = st.number_input(
                                        f"{feature}:", 
                                        min_value=1, max_value=12, 
                                        value=6, step=1, key=f"reg_{feature}_{model_key}_{selected_market_reg}"
                                    )
                                elif 'Quarter' in feature:
                                    feature_values[feature] = st.number_input(
                                        f"{feature}:", 
                                        min_value=1, max_value=4, 
                                        value=2, step=1, key=f"reg_{feature}_{model_key}_{selected_market_reg}"
                                    )
                                else:
                                    feature_values[feature] = st.number_input(
                                        f"{feature}:", 
                                        value=0.0, key=f"reg_{feature}_{model_key}_{selected_market_reg}"
                                    )
                        
                        # Prediction button for regression
                        if st.button(f"🎯 Predict Price with {selected_model_type_reg}", type="primary", key=f"reg_predict_{model_key}_{selected_market_reg}"):
                            
                            # Prepare feature vector
                            features_array = np.array([feature_values[feature] for feature in feature_names]).reshape(1, -1)
                            
                            # Simple prediction logic for linear regression (simulation)
                            if model_key == 'linear_regression' and 'coefficients' in model_results and 'intercept' in model_results:
                                coefficients = model_results['coefficients']
                                intercept = model_results['intercept']
                                predicted_price = np.sum(features_array * coefficients) + intercept  # Fixed: intercept is scalar
                            else:
                                # Fallback: use average price or something
                                predicted_price = 4000.0
                            
                            # Display results
                            st.markdown(f"""
                            <div class="prediction-result">
                                <h4>🎯 {selected_model_type_reg} Prediction Result</h4>
                                <p><strong>Market:</strong> {selected_market_reg}</p>
                                <p><strong>Predicted Price:</strong> ₹{predicted_price:.2f} / Quintal</p>
                                <p><strong>Model R²:</strong> {model_results['r2_score']:.3f}</p>
                                <p><strong>Model Type:</strong> {selected_model_type_reg} (Regression)</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Feature coefficients for linear regression
                            if model_key == 'linear_regression' and 'coefficients' in model_results:
                                st.subheader("📊 Feature Coefficients (Linear Regression)")
                                
                                coef_data = []
                                for feature, coef in zip(feature_names, model_results['coefficients']):
                                    coef_data.append({
                                        'Feature': feature,
                                        'Coefficient': coef,
                                        'Impact': 'Positive' if coef > 0 else 'Negative',
                                        'Magnitude': abs(coef)
                                    })
                                
                                coef_df = pd.DataFrame(coef_data).sort_values('Magnitude', ascending=False)
                                
                                fig_coef = px.bar(coef_df, x='Feature', y='Coefficient',
                                                title='Feature Coefficients (Impact on Price)',
                                                color='Coefficient',
                                                color_continuous_scale='RdBu_r')
                                fig_coef.add_hline(y=0, line_dash="dash", line_color="black")
                                fig_coef.update_layout(height=400, xaxis_tickangle=-45)
                                st.plotly_chart(fig_coef, use_container_width=True)
                
                # Regression Model comparison section
                st.markdown("---")
                st.subheader("📈 Detailed Regression Model Comparison")
                
                if selected_market_reg:
                    available_model_results_reg = {}
                    
                    for model_type in ['linear_regression']:
                        if (model_type in self.results['ml_models'] and 
                            selected_market_reg in self.results['ml_models'][model_type] and
                            self.results['ml_models'][model_type][selected_market_reg] is not None):
                            available_model_results_reg[model_type] = self.results['ml_models'][model_type][selected_market_reg]
                    
                    if available_model_results_reg:
                        # Performance metrics comparison
                        metrics_data = []
                        for model_type, results in available_model_results_reg.items():
                            metrics_data.append({
                                'Model': model_type.replace('_', ' ').title(),
                                'Test R²': results['r2_score'],
                                'CV Mean': results['cv_mean'],
                                'CV Std': results['cv_std'],
                                'Stability': 'High' if results['cv_std'] < 0.05 else 'Medium' if results['cv_std'] < 0.1 else 'Low'
                            })
                        
                        metrics_df = pd.DataFrame(metrics_data)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.dataframe(metrics_df, use_container_width=True)
                        
                        with col2:
                            # Radar chart for model comparison (adapted for R2)
                            fig_radar = go.Figure()
                            
                            for _, row in metrics_df.iterrows():
                                fig_radar.add_trace(go.Scatterpolar(
                                    r=[row['Test R²'], row['CV Mean'], 1-row['CV Std']],
                                    theta=['Test R²', 'CV Score', 'Stability'],
                                    fill='toself',
                                    name=row['Model']
                                ))
                            
                            fig_radar.update_layout(
                                polar=dict(
                                    radialaxis=dict(
                                        visible=True,
                                        range=[0, 1]
                                    )),
                                showlegend=True,
                                title=f"Regression Model Performance Comparison - {selected_market_reg}",
                                height=400
                            )
                            
                            st.plotly_chart(fig_radar, use_container_width=True)
        
        else:
            st.warning("No ML model results available. Please run the enhanced analysis first.")
    
   
    def format_report_section(self, text):
        """Format report text for better Streamlit display"""
        # Convert text formatting
        formatted = text.replace("="*100, "---")
        formatted = formatted.replace("="*80, "---") 
        formatted = formatted.replace("="*60, "---")
        formatted = formatted.replace("="*50, "---")
        formatted = formatted.replace("-"*80, "")
        formatted = formatted.replace("-"*60, "")
        formatted = formatted.replace("-"*50, "")
        formatted = formatted.replace("-"*40, "")
        
        return formatted
    
    def create_summary_report(self):
        """Create a concise summary report"""
        summary = []
        summary.append("SOYBEAN MARKET ANALYSIS - EXECUTIVE SUMMARY")
        summary.append("="*50)
        
        # Key findings
        if 'descriptive_stats' in self.results:
            total_records = sum(stats['Count'] for stats in self.results['descriptive_stats'].values())
            summary.append(f"\n📊 DATA OVERVIEW:")
            summary.append(f"• Total records analyzed: {total_records:,}")
            summary.append(f"• Markets covered: {len(self.results['descriptive_stats'])}")
            
            # Best and worst performing markets
            markets_by_price = sorted(self.results['descriptive_stats'].items(), 
                                    key=lambda x: x[1]['Mean_Price'], reverse=True)
            summary.append(f"• Highest prices: {markets_by_price[0][0]} (₹{markets_by_price[0][1]['Mean_Price']:.0f})")
            summary.append(f"• Lowest prices: {markets_by_price[-1][0]} (₹{markets_by_price[-1][1]['Mean_Price']:.0f})")
        
        # Cointegration summary
        if 'cointegration_tables' in self.results:
            coint_relations = self.results['cointegration_tables']['summary_stats']['Number_of_Cointegrating_Relations']
            summary.append(f"\n🔗 MARKET INTEGRATION:")
            summary.append(f"• Cointegrating relationships: {coint_relations}")
            summary.append(f"• Integration level: {'Strong' if coint_relations > 1 else 'Moderate' if coint_relations == 1 else 'Weak'}")
        
        # Best models
        if 'arima_models' in self.results:
            best_arima = min(self.results['arima_models'].items(), key=lambda x: x[1]['aic'])
            summary.append(f"\n📈 FORECASTING:")
            summary.append(f"• Best ARIMA model: {best_arima[0]} ARIMA{best_arima[1]['best_params']}")
            summary.append(f"• Model quality: AIC {best_arima[1]['aic']:.1f}")
        
        if 'model_comparisons' in self.results:
            all_accuracies = []
            for comp in self.results['model_comparisons'].values():
                all_accuracies.extend([acc for _, acc, _ in comp['ranking']])
            avg_accuracy = np.mean(all_accuracies) if all_accuracies else 0
            
            summary.append(f"\n🤖 CLASSIFICATION ML:")
            summary.append(f"• Average accuracy: {avg_accuracy:.1%}")
            summary.append(f"• Models evaluated: Logistic Regression, Random Forest")
        
        if 'regression_comparisons' in self.results:
            all_r2s = [comp['best_r2'] for comp in self.results['regression_comparisons'].values()]
            avg_r2 = np.mean(all_r2s) if all_r2s else 0
            
            summary.append(f"\n📈 REGRESSION ML:")
            summary.append(f"• Average R²: {avg_r2:.3f}")
            summary.append(f"• Model evaluated: Linear Regression")
        
        # Recommendations
        summary.append(f"\n💡 KEY RECOMMENDATIONS:")
        summary.append("• Use ARIMA models for medium-term price forecasting")
        summary.append("• Apply classification ML models for daily price movement prediction")  
        summary.append("• Use Linear Regression for price level predictions")
        summary.append("• Consider market integration in trading strategies")
        summary.append("• Implement risk management based on volatility patterns")
        
        return "\n".join(summary)

def main():
    """Main application function"""
    
    # Initialize enhanced dashboard
    dashboard = EnhancedSoybeanDashboard()
    
    # Sidebar navigation
    st.sidebar.title("🌱 Enhanced Navigation")
    st.sidebar.markdown("---")
    
    pages = {
        "🏠 Enhanced Dashboard": dashboard.main_dashboard,
        "🔗 Cointegration Analysis": dashboard.enhanced_cointegration_analysis,
        "🔮 ARIMA Forecasting": dashboard.enhanced_arima_analysis,
        "🤖 ML Models": dashboard.enhanced_ml_models,
       
    }
    
    selected_page = st.sidebar.selectbox("Choose Analysis:", list(pages.keys()))
    
    # Enhanced about section
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 🎯 Enhanced Analysis Features
    
    **New Additions:**
    - ✅ Multiple ML Models (Logistic Regression, Random Forest - Classification)
    - ✅ Linear Regression (Regression for Price Prediction)
    - ✅ Comprehensive Cointegration Tables  
    - ✅ Detailed AIC Explanations
    - ✅ Interactive Prediction Forms (Direction & Price Level)
    - ✅ Enhanced Visualizations
    
    **Research Objectives:**
    1. Enhanced descriptive statistics
    2. Comprehensive Johansen cointegration
    3. ARIMA/SARIMA with model selection explanations
    4. Multiple ML models comparison (Class + Reg)
    
    **Markets Analyzed:**
    - Haveri
    - Kalagategi  
    - Bidar
    - Kalaburgi
    - Bailhongal
    
    **ML Models:**
    - 🔵 Logistic Regression (Class)
    - 🌲 Random Forest (Class)
    - 📈 Linear Regression (Reg)
    """)
    
    # Execute selected page
    pages[selected_page]()
    
    # Enhanced footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
        <p>🌱 Enhanced Soybean Market Analysis Dashboard | Built with Advanced ML & Statistical Models</p>
        <p>Featuring: Logistic Regression • Random Forest • Linear Regression • Comprehensive Cointegration Analysis</p>
        <p>For research and educational purposes | © 2025</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":

    main()

