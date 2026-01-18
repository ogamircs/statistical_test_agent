
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from ab_agent import ABAgent

st.set_page_config(page_title="AntiGravity A/B Agent", layout="wide")

# Initialize Agent
if 'agent' not in st.session_state or not hasattr(st.session_state.agent, 'analyze_stratified_bayesian'):
    st.session_state.agent = ABAgent()

# --- SIDEBAR ---
st.sidebar.title("A/B Pilot Interaction")
uploaded_file = st.sidebar.file_uploader("Upload Experiment CSV", type=['csv'])

st.sidebar.divider()
st.sidebar.subheader("Statistical Thresholds")
alpha_input = st.sidebar.number_input("Significance Level (α)", min_value=0.001, max_value=0.500, value=0.05, step=0.01, format="%.3f", help="Threshold for P-value (Frequentist). Lower is stricter.")
bayes_threshold = st.sidebar.number_input("Bayesian Probability Threshold", min_value=0.500, max_value=0.999, value=0.95, step=0.01, format="%.3f", help="Probability required to declare a winner.")

if uploaded_file:
    # Load Data (Only once)
    if st.session_state.get('last_file') != uploaded_file.name:
        status, msg = st.session_state.agent.load_data(uploaded_file)
        if status:
            st.session_state.agent.guess_columns()
            st.session_state.agent.check_balance() # Initial check
            st.session_state.agent.check_pre_balance() # A/A check
            st.session_state.last_file = uploaded_file.name
            st.toast("Data loaded and analyzed!")
        else:
            st.error(f"Failed to load: {msg}")
            


# --- MAIN ---
st.title("🧪 AntiGravity A/B Testing Agent")

if st.session_state.agent.data is not None:
    df = st.session_state.agent.data
    mapping = st.session_state.agent.col_mapping
    
    # --- CHAT / FEEDBACK SECTION ---
    st.info("Agent Status Report")
    with st.expander("Agent Logs", expanded=True):
        for log in st.session_state.agent.report:
            st.text(f"🤖 {log}")
            
    # --- CONFIGURATION CHECK ---
    with st.container():
        st.subheader("1. Data Configuration")
        
        # Row 1: ID, Group, Segment
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            try: idx = df.columns.get_loc(mapping.get('id', df.columns[0]))
            except: idx = 0
            new_id = st.selectbox("ID Column", df.columns, index=idx)
        
        with r1c2:
            try: idx = df.columns.get_loc(mapping.get('group', df.columns[1]))
            except: idx = 0 if len(df.columns) > 0 else 0
            new_group = st.selectbox("Group Column", df.columns, index=idx)
            
        with r1c3:
            # Default segment selection
            default_seg = mapping.get('segment', [])
            if isinstance(default_seg, str): default_seg = [default_seg]
            # Ensure defaults are valid
            valid_seg = [c for c in default_seg if c in df.columns]
            
            new_segment = st.multiselect("Segment Column(s)", df.columns, default=valid_seg)

        # Row 2: Pre, Post (Metric)
        r2c1, r2c2 = st.columns(2)
        with r2c1:
            try: idx = df.columns.get_loc(mapping.get('pre_metric', df.columns[0]))
            except: idx = 0
            new_pre_metric = st.selectbox("Pre-Exp Metric (for DiD)", df.columns, index=idx)

        with r2c2:
            try: idx = df.columns.get_loc(mapping.get('metric', df.columns[-1]))
            except: idx = 0
            new_metric = st.selectbox("Target Metric (Post)", df.columns, index=idx)

        # Row 3: Covariates (Rest)
        # Filter available columns
        # Handle list vs singleton for used_cols
        used_cols = {new_id, new_group, new_pre_metric, new_metric}
        for s in new_segment: used_cols.add(s)
        
        available_covariates = [c for c in df.columns if c not in used_cols]
        # Current selection might contain items now used elsewhere, so filter default too
        current_feats = mapping.get('features', [])
        valid_default = [c for c in current_feats if c in available_covariates]
        
        new_feats = st.multiselect("Covariates (for balance)", available_covariates, default=valid_default)

        if st.button("Update Configuration & Re-Run"):
            st.session_state.agent.col_mapping = {
                'id': new_id, 'group': new_group, 'metric': new_metric, 
                'features': new_feats, 'segment': new_segment, 'pre_metric': new_pre_metric
            }
            # Clear previous logs to avoid clutter
            st.session_state.agent.report = [] 
            st.session_state.agent.log(f"Configuration updated manually.")
            st.session_state.agent.check_balance()
            st.session_state.agent.check_pre_balance() # Check A/A test
            if not st.session_state.agent.is_balanced and not st.session_state.agent.use_did:
                 st.session_state.agent.rebalance() # Only IPW if not using DiD (or combined, but keeping simple)
            st.rerun()

    # --- IMBALANCE ALERT ---
    if not st.session_state.agent.is_balanced:
        st.warning("⚠️ Imbalance detected in covariates! The agent has automatically applied Re-balancing weights (IPW) for the analysis below.")
    else:
        st.success("✅ Groups appear balanced on covariates.")

    # --- RESULTS DASHBOARD ---
    st.divider()
    st.subheader("2. Statistical Results")
    
    # Run Analysis
    freq_res = st.session_state.agent.analyze_frequentist(alpha=alpha_input)
    bayes_posteriors, bayes_comp = st.session_state.agent.analyze_bayesian()
    strat_res = st.session_state.agent.analyze_stratified(alpha=alpha_input)
    strat_bayes_res, strat_bayes_plots = st.session_state.agent.analyze_stratified_bayesian()
    
    t1, t2, t3, t4, t5 = st.tabs(["Frequentist (Classical)", "Bayesian (Probabilistic)", "Segment Analysis (Frequentist)", "Segment Analysis (Bayesian)", "Overall Comparison"])
    
    with t1:
        st.markdown("#### Frequentist Results")
        res_df = pd.DataFrame(freq_res).T
        if not res_df.empty:
            st.dataframe(res_df.style.applymap(lambda x: "background-color: #d4edda" if x == True else "", subset=['significant']))
            
            if st.session_state.agent.use_did:
                st.info("ℹ️ Using Difference-in-Differences (DiD) because pre-experiment imbalance was detected.")
                
        else:
            st.write("No results capable. Check config.")
            
        if not res_df.empty:
             fig = px.bar(res_df, y='relative_effect', x='treatment', title="Relative Effect Size vs Control", 
                          color='significant', text_auto='.2%', hover_data=['method'])
             st.plotly_chart(fig)

    with t2:
        st.markdown("#### Posterior Distributions")
        
        # Plot distributions
        fig_bayes = go.Figure()
        
        # Dynamic X-range based on means (matching segment analysis logic)
        all_means = [res.get('mean', 0.5) for res in bayes_posteriors.values()]
        if all_means:
            min_mean, max_mean = min(all_means), max(all_means)
            # Handle negative means correctly for "zoom"
            x_min = min_mean * 0.8 if min_mean >= 0 else min_mean * 1.2
            x_max = max_mean * 1.2 if max_mean >= 0 else max_mean * 0.8
            if x_min == x_max: x_min, x_max = x_min - 0.5, x_max + 0.5
            x_range = np.linspace(x_min, x_max, 1000)
        else:
            x_range = np.linspace(df[new_metric].min(), df[new_metric].max(), 1000)
        
        for g, res in bayes_posteriors.items():
            if res['type'] == 'normal_t':
                y = stats.t.pdf(x_range, res['df'], res['mean'], res['std_err'])
                fig_bayes.add_trace(go.Scatter(x=x_range, y=y, name=f"{g} (Mean={res['mean']:.2f})", fill='tozeroy', opacity=0.5))
            elif res['type'] == 'beta':
                # For beta, x range is 0-1
                x_b = np.linspace(0, 1, 500)
                y = stats.beta.pdf(x_b, res['alpha'], res['beta'])
                fig_bayes.add_trace(go.Scatter(x=x_b, y=y, name=f"{g} (Rate={res['mean']:.2%})", fill='tozeroy', opacity=0.5))
                
        st.plotly_chart(fig_bayes, use_container_width=True)
        
        st.markdown("#### Decision Support")
        for g, res in bayes_comp.items():
            prob = res['prob_treatment_better']
            is_sig = prob > bayes_threshold
            
            w_str = "✅ **WINNER**" if is_sig else "Uncertain"
            st.metric(f"Probability {g} > Control", f"{prob:.1%}", 
                      delta=f"Exp. Uplift: {res['expected_uplift']:.4f}")
            if is_sig:
                st.success(f"With {prob:.1%} probability, {g} is better than Control (Threshold: {bayes_threshold:.0%})")
            else:
                st.caption(f"Requires > {bayes_threshold:.0%} probability to be significant.")

    with t3:
        st.markdown(f"#### Frequentist Analysis by {mapping.get('segment', 'Segment')}")
        if strat_res is not None and not strat_res.empty:
            # Highlight total row
            def highlight_total(row):
                if row['segment_value'] == 'TOTAL':
                    return ['font-weight: bold; background-color: #f0f2f6'] * len(row)
                return [''] * len(row)

            # Drop unnecessary columns for display to match Bayesian view
            display_df = strat_res.drop(columns=['treatment', 'segment_col'], errors='ignore')
            
            st.dataframe(display_df.style.apply(highlight_total, axis=1)
                         .format({'effect_size': '{:.4f}', 'p_value': '{:.4f}', 'aa_p_val': '{:.4f}', 'weight': '{:.0f}'}))
            
            # Plot segments
            seg_only = strat_res[strat_res['segment_value'] != 'TOTAL']
            if not seg_only.empty:
                fig_seg = px.bar(seg_only, x='segment_value', y='effect_size', color='treatment', 
                                 title="Effect Size by Segment (Frequentist)", error_y=None, barmode='group')
                st.plotly_chart(fig_seg)
        else:
            st.warning("Could not perform stratified analysis. Check segment column.")

    with t4:
        st.markdown(f"#### Bayesian Analysis by {mapping.get('segment', 'Segment')}")
        if strat_bayes_res is not None and not strat_bayes_res.empty:
             def highlight_total_bayes(row):
                if row['segment_value'] == 'TOTAL (Weighted)':
                    return ['font-weight: bold; background-color: #e6f3ff'] * len(row)
                return [''] * len(row)

             st.dataframe(strat_bayes_res.style.apply(highlight_total_bayes, axis=1)
                          .format({'prob_treatment_better': '{:.2%}', 'expected_uplift': '{:.4f}', 
                                   'control_mean': '{:.4f}', 'treatment_mean': '{:.4f}', 
                                   'control_support': '{}', 'treatment_support': '{}', 'total_support': '{}'}))
             
             st.divider()
             st.markdown("#### Segment Deep Dive: Posterior Distributions")
             
             # Segment Selector
             unique_strats = list(strat_bayes_plots.keys())
             # Default to TOTAL if exists
             default_idx = unique_strats.index('TOTAL') if 'TOTAL' in unique_strats else 0
             selected_seg = st.selectbox("Select Segment to Visualize", unique_strats, index=default_idx)
             
             if selected_seg:
                 params_dict = strat_bayes_plots[selected_seg]
                 fig_strat_bayes = go.Figure()
                 
                 # Dynamic X-range based on means in this segment
                 all_means = [p.get('mean', 0.5) for p in params_dict.values()]
                 min_x, max_x = min(all_means) * 0.8, max(all_means) * 1.2
                 if min_x == max_x: min_x, max_x = 0, 1
                 
                 x_vals = np.linspace(min_x, max_x, 1000)
                 
                 for g, params in params_dict.items():
                     if params['type'] == 'normal_t':
                         y = stats.t.pdf(x_vals, params['df'], params['mean'], params['std_err'])
                         fig_strat_bayes.add_trace(go.Scatter(x=x_vals, y=y, name=f"{g} (Mean={params['mean']:.2f})", fill='tozeroy', opacity=0.5))
                     elif params['type'] == 'beta':
                         x_b = np.linspace(0, 1, 1000)
                         y = stats.beta.pdf(x_b, params['alpha'], params['beta'])
                         fig_strat_bayes.add_trace(go.Scatter(x=x_b, y=y, name=f"{g} (Rate={params['mean']:.2%})", fill='tozeroy', opacity=0.5))
                         
                 fig_strat_bayes.update_layout(title=f"Posterior Distributions: {selected_seg}", xaxis_title=new_metric, yaxis_title="Density")
                 st.plotly_chart(fig_strat_bayes, use_container_width=True)
                 
        else:
            st.warning("Could not perform Bayesian segment analysis.")

    with t5:
        st.markdown("#### Head-to-Head Comparison")
        st.markdown("Comparison of overall effect estimates between Frequentist and Bayesian methods.")
        
        comp_rows = []
        for g, f_data in freq_res.items():
            b_data = bayes_comp.get(g, {})
            
            row = {
                'Treatment Group': g,
                'Freq Effect': f_data.get('effect_size'),
                'Freq P-Value': f_data.get('p_value'),
                'Bayes Uplift': b_data.get('expected_uplift'),
                'Bayes Prob > Control': b_data.get('prob_treatment_better'),
                'Bayes Significant': b_data.get('prob_treatment_better', 0) > bayes_threshold
            }
            comp_rows.append(row)
            
        if comp_rows:
            comp_df = pd.DataFrame(comp_rows)
            st.dataframe(comp_df.style.format({
                'Freq Effect': '{:.4f}',
                'Freq P-Value': '{:.4f}', 
                'Bayes Uplift': '{:.4f}',
                'Bayes Prob > Control': '{:.2%}'
            }))
            
            # Visual comparison of effect sizes
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(name='Freq Effect', x=comp_df['Treatment Group'], y=comp_df['Freq Effect'], text=comp_df['Freq Effect'].apply(lambda x: f'{x:.4f}'), textposition='auto'))
            fig_comp.add_trace(go.Bar(name='Bayes Uplift', x=comp_df['Treatment Group'], y=comp_df['Bayes Uplift'], text=comp_df['Bayes Uplift'].apply(lambda x: f'{x:.4f}'), textposition='auto'))
            fig_comp.update_layout(title="Effect Size Comparison", barmode='group')
            st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.write("No results to compare.")

else:
    st.info("👈 Please upload a CSV file to begin.")
    st.markdown("""
    ### Example Format
    | user_id | group | revenue | age_group | region |
    |---|---|---|---|---|
    | 123 | test | 45.2 | 25-34 | US |
    | 124 | control | 0.0 | 18-24 | EU |
    """)

