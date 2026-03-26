import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import google.generativeai as genai
import os

st.set_page_config(page_title="Acceptance Rate Dashboard", layout="wide")

# --- FILE NAMES ---
# Make sure these match the exact names of the CSV files you uploaded to your GitHub repo!
AR_FILE = "aggregated_ar_data.csv"
FAIL_FILE = "aggregated_failure_data.csv"
ISSUER_FILE = "aggregated_issuer_data.csv.gz"

# --- 1. DATA LOADING (AUTOMATED FROM GITHUB REPO) ---
st.title("Acceptance Rate Performance Dashboard")

# Initialize memory banks
if 'core_df' not in st.session_state:
    st.session_state.core_df = None
if 'core_fail_df' not in st.session_state:
    st.session_state.core_fail_df = None

def load_local_data(filepath):
    if os.path.exists(filepath):
        try:
            temp_df = pd.read_csv(filepath, sep='\t')
            if len(temp_df.columns) < 5:
                temp_df = pd.read_csv(filepath)
            return temp_df
        except Exception as e:
            st.error(f"Error reading {filepath}: {e}")
            return None
    else:
        st.warning(f"⚠️ Could not find {filepath} in the repository. Please upload it to GitHub.")
        return None

# Load files automatically into memory
if st.session_state.core_df is None:
    st.session_state.core_df = load_local_data(AR_FILE)
if st.session_state.core_fail_df is None:
    st.session_state.core_fail_df = load_local_data(FAIL_FILE)

# Pull from Memory!
df = st.session_state.core_df
fail_df = st.session_state.core_fail_df

# Check if we have data to proceed
if df is None:
    st.stop()
else:
    st.success("✅ Dashboard data loaded automatically.")

# --- 2. DATA PREPARATION ---
try:
    df = df.fillna(0)
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Month'] = df['Order Date'].dt.to_period('M').astype(str)
    for col in ['Success Order Count', 'Recovery Order Count', 'Order Count']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    df['Acceptance Orders'] = df['Success Order Count'] + df['Recovery Order Count']
    
    if fail_df is not None:
        fail_df = fail_df.fillna(0)
        fail_df['Order Date'] = pd.to_datetime(fail_df['Order Date'])
        fail_df['Month'] = fail_df['Order Date'].dt.to_period('M').astype(str)
        fail_df['Fail Order Count'] = pd.to_numeric(fail_df['Fail Order Count'], errors='coerce').fillna(0)

    # --- SPLIT PANDORA INTO BRANDS ---
    brand_mapping = {
        'Taiwan': 'Foodpanda', 'Malaysia': 'Foodpanda', 'Hong Kong': 'Foodpanda', 
        'Singapore': 'Foodpanda', 'Philippines': 'Foodpanda', 'Pakistan': 'Foodpanda', 
        'Bangladesh': 'Foodpanda', 'Cambodia': 'Foodpanda', 'Myanmar': 'Foodpanda', 'Laos': 'Foodpanda',
        'Sweden': 'Foodora', 'Norway': 'Foodora', 'Austria': 'Foodora', 
        'Hungary': 'Foodora', 'Czech Republic': 'Foodora',
        'Turkey': 'Yemeksepeti'
    }

    if 'data_source' in df.columns and 'Country' in df.columns:
        is_pandora = df['data_source'] == 'Pandora'
        df.loc[is_pandora, 'data_source'] = df.loc[is_pandora, 'Country'].map(brand_mapping).fillna('Pandora')

    if fail_df is not None and 'data_source' in fail_df.columns and 'Country' in fail_df.columns:
        is_pandora_fail = fail_df['data_source'] == 'Pandora'
        fail_df.loc[is_pandora_fail, 'data_source'] = fail_df.loc[is_pandora_fail, 'Country'].map(brand_mapping).fillna('Pandora')
        
except Exception as e:
    st.error(f"🚨 Data Preparation Error: {e}")
    st.stop()

months = sorted(df['Month'].unique())
if len(months) < 2:
    st.info("Please upload data with at least two distinct months to view MoM attribution.")
    st.stop()

curr_month, prev_month = months[-1], months[-2]

# --- REUSABLE VARIANCE MATH ---
def calculate_variance(sub_df_curr, sub_df_prev, join_keys):
    merged = pd.merge(sub_df_curr, sub_df_prev, on=join_keys, suffixes=('_curr', '_prev'), how='outer').fillna(0)
    oc = merged['Order Count_curr'].sum()
    op = merged['Order Count_prev'].sum()
    if oc == 0 or op == 0: return merged, 0, 0, 0
    
    arc = merged['Acceptance Orders_curr'].sum() / oc
    arp = merged['Acceptance Orders_prev'].sum() / op
    total_delta = arc - arp
    
    merged['W_curr'] = merged['Order Count_curr'] / oc
    merged['W_prev'] = merged['Order Count_prev'] / op
    
    merged['AR_curr_item'] = np.where(merged['Order Count_curr']>0, merged['Acceptance Orders_curr']/merged['Order Count_curr'], 0)
    merged['AR_prev_item'] = np.where(merged['Order Count_prev']>0, merged['Acceptance Orders_prev']/merged['Order Count_prev'], 0)
    
    # Handle new PMs by assuming they started at the previous global average
    merged['AR_prev_adj'] = np.where(merged['Order Count_prev']==0, arp, merged['AR_prev_item'])
    
    merged['Mix_Impact'] = (merged['W_curr'] - merged['W_prev']) * merged['AR_prev_adj']
    merged['Rate_Impact'] = merged['W_curr'] * (merged['AR_curr_item'] - merged['AR_prev_adj'])
    
    return merged, total_delta, merged['Rate_Impact'].sum(), merged['Mix_Impact'].sum()

def get_highlighter(threshold, max_scale, reverse_colors=False):
    def highlight(val):
        if not isinstance(val, (int, float)) or pd.isna(val): return ''
        if val >= threshold:
            intensity = min(val / max_scale, 1) 
            color = '231, 76, 60' if reverse_colors else '39, 174, 96'
            return f'background-color: rgba({color}, {0.15 + (0.4 * intensity)});'
        elif val <= -threshold:
            intensity = min(abs(val) / max_scale, 1)
            color = '39, 174, 96' if reverse_colors else '231, 76, 60'
            return f'background-color: rgba({color}, {0.15 + (0.4 * intensity)});'
        return ''
    return highlight

# --- 3. STATIC GLOBAL PERFORMANCE HEADER ---
st.divider()
st.header(f"🌍 Global Performance: {curr_month} vs {prev_month}")
st.markdown("*(These values represent the entire unfiltered dataset)*")

global_toggle = st.radio("Calculate Global Drivers by:", ["All Payment Methods", "CC vs APM"], horizontal=True)

# Math for the static header
agg_cols = ['Order Count', 'Acceptance Orders']
join_keys = ['data_source', 'First Payment Method'] if global_toggle == "All Payment Methods" else ['data_source', 'CC VS APM']

# Get the raw monthly data
df_month_curr = df[df['Month'] == curr_month]
df_month_prev = df[df['Month'] == prev_month]

# Aggregate globally
df_global_curr = df_month_curr.groupby(join_keys)[agg_cols].sum().reset_index()
df_global_prev = df_month_prev.groupby(join_keys)[agg_cols].sum().reset_index()

# Calculate global totals
global_merged, g_delta, g_perf, g_mix = calculate_variance(df_global_curr, df_global_prev, join_keys)
global_ar = df_global_curr['Acceptance Orders'].sum() / df_global_curr['Order Count'].sum() if df_global_curr['Order Count'].sum() > 0 else 0

# KPI METRICS BLOCK
if global_toggle == "All Payment Methods":
    col1, col2, col3 = st.columns(3)
    col1.metric("Global AR (Latest)", f"{global_ar:.2%}", f"{g_delta:+.2%}")
    col2.metric("Global Performance Driver", f"{g_perf:+.2%}")
    col3.metric("Global Mix Driver", f"{g_mix:+.2%}")
else:
    col1, col2, col3 = st.columns(3)
    col1.metric("Global AR (Latest)", f"{global_ar:.2%}", f"{g_delta:+.2%}")
    col2.metric("Global Performance Driver", f"{g_perf:+.2%}")
    col3.metric("Global Mix Driver", f"{g_mix:+.2%}")
    
    st.write("") 
    
    cc_curr = df_month_curr[df_month_curr['CC VS APM'] == 'Credit Card'].groupby(join_keys)[agg_cols].sum().reset_index()
    cc_prev = df_month_prev[df_month_prev['CC VS APM'] == 'Credit Card'].groupby(join_keys)[agg_cols].sum().reset_index()
    _, cc_delta, cc_perf, cc_mix = calculate_variance(cc_curr, cc_prev, join_keys)
    cc_ar = cc_curr['Acceptance Orders'].sum() / cc_curr['Order Count'].sum() if cc_curr['Order Count'].sum() > 0 else 0
    
    apm_curr = df_month_curr[df_month_curr['CC VS APM'] == 'Alternative Payment Method'].groupby(join_keys)[agg_cols].sum().reset_index()
    apm_prev = df_month_prev[df_month_prev['CC VS APM'] == 'Alternative Payment Method'].groupby(join_keys)[agg_cols].sum().reset_index()
    _, apm_delta, apm_perf, apm_mix = calculate_variance(apm_curr, apm_prev, join_keys)
    apm_ar = apm_curr['Acceptance Orders'].sum() / apm_curr['Order Count'].sum() if apm_curr['Order Count'].sum() > 0 else 0

    cc_col, apm_col = st.columns(2)
    with cc_col:
        st.markdown("#### 💳 Credit Cards")
        sub1, sub2, sub3 = st.columns(3)
        sub1.metric("CC AR", f"{cc_ar:.2%}", f"{cc_delta:+.2%}")
        sub2.metric("Performance Impact", f"{cc_perf:+.2%}")
        sub3.metric("Mix Impact", f"{cc_mix:+.2%}")
        
    with apm_col:
        st.markdown("#### 📱 APMs")
        sub1, sub2, sub3 = st.columns(3)
        sub1.metric("APM AR", f"{apm_ar:.2%}", f"{apm_delta:+.2%}")
        sub2.metric("Performance Impact", f"{apm_perf:+.2%}")
        sub3.metric("Mix Impact", f"{apm_mix:+.2%}")

# TABLE DISPLAY BLOCK (Rolled up perfectly by Entity)
display_keys = ['data_source'] if global_toggle == "All Payment Methods" else ['data_source', 'CC VS APM']

entity_summary = global_merged.groupby(display_keys).agg({
    'Order Count_curr': 'sum', 'Order Count_prev': 'sum',
    'Acceptance Orders_curr': 'sum', 'Acceptance Orders_prev': 'sum',
    'Rate_Impact': 'sum', 'Mix_Impact': 'sum'
}).reset_index()

entity_summary['Latest AR'] = np.where(entity_summary['Order Count_curr']>0, entity_summary['Acceptance Orders_curr']/entity_summary['Order Count_curr'], 0)
entity_summary['Prev AR'] = np.where(entity_summary['Order Count_prev']>0, entity_summary['Acceptance Orders_prev']/entity_summary['Order Count_prev'], 0)
entity_summary['MoM Delta'] = entity_summary['Latest AR'] - entity_summary['Prev AR']

display_cols = display_keys + ['Latest AR', 'MoM Delta', 'Rate_Impact', 'Mix_Impact']
global_display = entity_summary[display_cols].sort_values(by=display_keys).rename(
    columns={'data_source': 'Entity', 'Rate_Impact': 'Global Rate Impact', 'Mix_Impact': 'Global Mix Impact'}
)

format_cols = ['Latest AR', 'MoM Delta', 'Global Rate Impact', 'Global Mix Impact']

st.dataframe(
    global_display.style.format({c: "{:.2%}" for c in format_cols})
    .map(get_highlighter(0.001, 0.05), subset=['MoM Delta', 'Global Rate Impact', 'Global Mix Impact']),
    use_container_width=True, hide_index=True
)

# --- 4. CASCADING FILTERS (APPLIES TO EVERYTHING BELOW) ---
st.divider()
st.sidebar.header("Filters (Applies to Deep Dive)")
metric_choice = st.sidebar.selectbox("Select Metric for Trend Chart", ['Acceptance Rate', 'Success Rate', 'Recovery Rate'])

try:
    if 'data_source' in df.columns:
        sel_source = st.sidebar.selectbox("Data Source", ['All'] + list(df['data_source'].astype(str).unique()))
        if sel_source != 'All': 
            df = df[df['data_source'] == sel_source]
            if fail_df is not None and 'data_source' in fail_df.columns: fail_df = fail_df[fail_df['data_source'] == sel_source]
            
    entity_df = df.copy() # Freeze unfiltered entity data for accurate GT weights
        
    if 'Country' in df.columns:
        sel_country = st.sidebar.selectbox("Country", ['All'] + list(df['Country'].astype(str).unique()))
        if sel_country != 'All': 
            df = df[df['Country'] == sel_country]
            if fail_df is not None and 'Country' in fail_df.columns: fail_df = fail_df[fail_df['Country'] == sel_country]
            
    if 'CC VS APM' in df.columns:
        sel_cc = st.sidebar.selectbox("CC VS APM", ['All'] + list(df['CC VS APM'].astype(str).unique()))
        if sel_cc != 'All': 
            df = df[df['CC VS APM'] == sel_cc]
            if fail_df is not None and 'CC VS APM' in fail_df.columns: fail_df = fail_df[fail_df['CC VS APM'] == sel_cc]
            
    if 'First Payment Method' in df.columns:
        sel_pm = st.sidebar.selectbox("First Payment Method", ['All'] + list(df['First Payment Method'].astype(str).unique()))
        if sel_pm != 'All': 
            df = df[df['First Payment Method'] == sel_pm]
            if fail_df is not None and 'First Payment Method' in fail_df.columns: fail_df = fail_df[fail_df['First Payment Method'] == sel_pm]
except Exception as e:
    st.warning(f"Filter error: {e}")

# --- 5. FILTERED DEEP DIVE SECTION ---
st.header("🔍 Filtered Deep Dive")

# Trend Chart
trend_df = df.groupby('Month')[['Order Count', 'Success Order Count', 'Recovery Order Count', 'Acceptance Orders']].sum().reset_index()
trend_df['Success Rate'] = np.where(trend_df['Order Count'] > 0, trend_df['Success Order Count'] / trend_df['Order Count'], 0)
trend_df['Recovery Rate'] = np.where(trend_df['Order Count'] > 0, trend_df['Recovery Order Count'] / trend_df['Order Count'], 0)
trend_df['Acceptance Rate'] = np.where(trend_df['Order Count'] > 0, trend_df['Acceptance Orders'] / trend_df['Order Count'], 0)
fig = px.line(trend_df, x='Month', y=metric_choice, markers=True, title=f"Monthly {metric_choice} Trend")
fig.update_layout(yaxis_tickformat='.2%')
st.plotly_chart(fig, use_container_width=True)

# Attribution Math Base
agg_cols_ext = ['Order Count', 'Success Order Count', 'Recovery Order Count', 'Acceptance Orders']
df_curr = df[df['Month'] == curr_month].groupby(['data_source', 'Country', 'CC VS APM', 'First Payment Method'])[agg_cols_ext].sum().reset_index()
df_prev = df[df['Month'] == prev_month].groupby(['data_source', 'Country', 'CC VS APM', 'First Payment Method'])[agg_cols_ext].sum().reset_index()
merged = pd.merge(df_curr, df_prev, on=['data_source', 'Country', 'CC VS APM', 'First Payment Method'], suffixes=('_curr', '_prev'), how='outer').fillna(0)

# Local Variance Analysis
def get_local_variance(sub_df):
    oc, op = sub_df['Order Count_curr'].sum(), sub_df['Order Count_prev'].sum()
    if oc == 0 or op == 0: return 0, 0, 0
    arc, arp = sub_df['Acceptance Orders_curr'].sum() / oc, sub_df['Acceptance Orders_prev'].sum() / op
    delta = arc - arp
    wc, wp = sub_df['Order Count_curr'] / oc, sub_df['Order Count_prev'] / op
    ar_c_item = np.where(sub_df['Order Count_curr']>0, sub_df['Acceptance Orders_curr']/sub_df['Order Count_curr'], 0)
    ar_p_item = np.where(sub_df['Order Count_prev']>0, sub_df['Acceptance Orders_prev']/sub_df['Order Count_prev'], 0)
    ar_p_adj = np.where(sub_df['Order Count_prev']==0, arp, ar_p_item)
    perf = (wc * (ar_c_item - ar_p_adj)).sum()
    mix = ((wc - wp) * ar_p_adj).sum()
    return delta, perf, mix

tot_del, tot_perf, tot_mix = get_local_variance(merged)
cc_del, cc_perf, cc_mix = get_local_variance(merged[merged['CC VS APM'] == 'Credit Card'])
apm_del, apm_perf, apm_mix = get_local_variance(merged[merged['CC VS APM'] == 'Alternative Payment Method'])

st.subheader(f"📊 Filtered Variance: {curr_month} vs {prev_month}")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("#### 🌐 Total (Filtered)")
    st.metric("Total AR Change", f"{tot_del:+.2%}")
    st.metric("Performance Impact", f"{tot_perf:+.2%}")
    st.metric("Mix Impact", f"{tot_mix:+.2%}")
with c2:
    st.markdown("#### 💳 Credit Cards")
    st.metric("CC AR Change", f"{cc_del:+.2%}")
    st.metric("CC Performance Impact", f"{cc_perf:+.2%}")
    st.metric("CC Mix Impact", f"{cc_mix:+.2%}")
with c3:
    st.markdown("#### 📱 APMs")
    st.metric("APM AR Change", f"{apm_del:+.2%}")
    st.metric("APM Performance Impact", f"{apm_perf:+.2%}")
    st.metric("APM Mix Impact", f"{apm_mix:+.2%}")

# Attribution Table
st.subheader("Performance vs Last Month (Mix & Yield Attribution)")
merged['AR_curr'] = np.where(merged['Order Count_curr'] > 0, merged['Acceptance Orders_curr'] / merged['Order Count_curr'], 0)
merged['AR_prev'] = np.where(merged['Order Count_prev'] > 0, merged['Acceptance Orders_prev'] / merged['Order Count_prev'], 0)
merged['MoM_Delta'] = merged['AR_curr'] - merged['AR_prev']

merged['Country_Orders_curr'] = merged.groupby(['data_source', 'Country'])['Order Count_curr'].transform('sum')
merged['Country_Orders_prev'] = merged.groupby(['data_source', 'Country'])['Order Count_prev'].transform('sum')
merged['W_sub_curr'] = np.where(merged['Country_Orders_curr'] > 0, merged['Order Count_curr'] / merged['Country_Orders_curr'], 0)
merged['W_sub_prev'] = np.where(merged['Country_Orders_prev'] > 0, merged['Order Count_prev'] / merged['Country_Orders_prev'], 0)

entity_totals_curr = entity_df[entity_df['Month'] == curr_month].groupby('data_source')['Order Count'].sum().reset_index(name='GT_Orders_curr')
entity_totals_prev = entity_df[entity_df['Month'] == prev_month].groupby('data_source')['Order Count'].sum().reset_index(name='GT_Orders_prev')
merged = pd.merge(merged, entity_totals_curr, on='data_source', how='left').fillna(0)
merged = pd.merge(merged, entity_totals_prev, on='data_source', how='left').fillna(0)

merged['W_gt_curr'] = np.where(merged['GT_Orders_curr'] > 0, merged['Order Count_curr'] / merged['GT_Orders_curr'], 0)
merged['W_gt_prev'] = np.where(merged['GT_Orders_prev'] > 0, merged['Order Count_prev'] / merged['GT_Orders_prev'], 0)
merged['Subtotal_Mix_Impact'] = (merged['W_sub_curr'] - merged['W_sub_prev']) * merged['AR_prev']
merged['Subtotal_Rate_Impact'] = merged['W_sub_curr'] * merged['MoM_Delta']
merged['GT_Mix_Impact'] = (merged['W_gt_curr'] - merged['W_gt_prev']) * merged['AR_prev']
merged['GT_Rate_Impact'] = merged['W_gt_curr'] * merged['MoM_Delta']

display_df = merged[['data_source', 'Country', 'First Payment Method', 'AR_curr', 'MoM_Delta', 
                     'Subtotal_Mix_Impact', 'Subtotal_Rate_Impact', 'GT_Mix_Impact', 'GT_Rate_Impact']].copy()
display_df.rename(columns={'data_source': 'Entity', 'AR_curr': 'Latest AR', 'MoM_Delta': 'MoM Delta',
                           'Subtotal_Mix_Impact': 'Mix Impact (Country)', 'Subtotal_Rate_Impact': 'Rate Impact (Country)',
                           'GT_Mix_Impact': 'Mix Impact (Entity)', 'GT_Rate_Impact': 'Rate Impact (Entity)'}, inplace=True)
display_df = display_df.sort_values(by=['Entity', 'Country', 'First Payment Method'])

country_cols = [col for col in display_df.columns if '(Country)' in col or 'Delta' in col]
entity_cols = [col for col in display_df.columns if '(Entity)' in col]

st.dataframe(
    display_df.style.format({col: "{:.2%}" for col in display_df.columns if 'AR' in col or 'Delta' in col or 'Impact' in col})
    .map(get_highlighter(0.001, 0.05), subset=country_cols)
    .map(get_highlighter(0.0001, 0.01), subset=entity_cols),
    height=400, use_container_width=True, hide_index=True
)

# AR Breakdown Table (Success vs Recovery)
st.subheader("Breakdown of Acceptance Rate (Success vs. Recovery)")
brk_df = merged[['data_source', 'Country', 'CC VS APM', 'First Payment Method']].copy()
brk_df['Total AR Delta'] = np.where(merged['Order Count_curr']>0, merged['Acceptance Orders_curr']/merged['Order Count_curr'], 0) - np.where(merged['Order Count_prev']>0, merged['Acceptance Orders_prev']/merged['Order Count_prev'], 0)
brk_df['Success Delta'] = np.where(merged['Order Count_curr']>0, merged['Success Order Count_curr']/merged['Order Count_curr'], 0) - np.where(merged['Order Count_prev']>0, merged['Success Order Count_prev']/merged['Order Count_prev'], 0)
brk_df['Recovery Delta'] = np.where(merged['Order Count_curr']>0, merged['Recovery Order Count_curr']/merged['Order Count_curr'], 0) - np.where(merged['Order Count_prev']>0, merged['Recovery Order Count_prev']/merged['Order Count_prev'], 0)

brk_df.rename(columns={'data_source': 'Entity'}, inplace=True)
st.dataframe(
    brk_df.sort_values(by=['Entity', 'Country', 'First Payment Method']).style.format({c: "{:.2%}" for c in brk_df.columns if 'Delta' in c})
    .map(get_highlighter(0.001, 0.05), subset=['Total AR Delta', 'Success Delta', 'Recovery Delta']),
    height=400, use_container_width=True, hide_index=True
)

# --- 6. TOP DECLINE REASONS (NEW LOGIC) ---
if fail_df is not None:
    st.divider()
    st.subheader(f"🛑 Payment Decline Drivers ({curr_month} vs {prev_month})")
    
    total_orders_curr = df_curr['Order Count'].sum()
    total_orders_prev = df_prev['Order Count'].sum()
    
    fail_curr_df = fail_df[fail_df['Month'] == curr_month].groupby(['Psp Operation Status', 'Psp Actionability'])['Fail Order Count'].sum().reset_index()
    fail_prev_df = fail_df[fail_df['Month'] == prev_month].groupby(['Psp Operation Status', 'Psp Actionability'])['Fail Order Count'].sum().reset_index()
    
    total_failures_curr = fail_curr_df['Fail Order Count'].sum()
    
    fail_merged = pd.merge(fail_curr_df, fail_prev_df, on=['Psp Operation Status', 'Psp Actionability'], suffixes=('_curr', '_prev'), how='outer').fillna(0)
    
    if total_orders_curr > 0 and total_orders_prev > 0:
        fail_merged['% of Total Orders (Curr)'] = fail_merged['Fail Order Count_curr'] / total_orders_curr
        fail_merged['% of Total Orders (Prev)'] = fail_merged['Fail Order Count_prev'] / total_orders_prev
        fail_merged['MoM Delta (% of Orders)'] = fail_merged['% of Total Orders (Curr)'] - fail_merged['% of Total Orders (Prev)']
        fail_merged['% of Total Failures'] = np.where(total_failures_curr > 0, fail_merged['Fail Order Count_curr'] / total_failures_curr, 0)
        
        fail_display = fail_merged[fail_merged['Fail Order Count_curr'] > 0].sort_values(by='% of Total Orders (Curr)', ascending=False).head(20)
        
        col_chart, col_table = st.columns([1, 1.5])
        
        with col_chart:
            fig_fail = px.bar(
                fail_display.head(10).sort_values(by='% of Total Orders (Curr)', ascending=True), 
                x='% of Total Orders (Curr)', 
                y='Psp Operation Status', 
                color='Psp Actionability', 
                orientation='h',
                text='% of Total Orders (Curr)',
                color_discrete_map={'Actionable': '#f39c12', 'Non-actionable': '#e74c3c', 'unknown': '#95a5a6'},
                title="Top Decline Reasons (% of Total Orders)"
            )
            fig_fail.update_traces(texttemplate='%{text:.2%}', textposition='outside')
            fig_fail.update_layout(xaxis_tickformat='.1%', showlegend=False, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_fail, use_container_width=True)
            
        with col_table:
            display_fail_table = fail_display[['Psp Operation Status', 'Psp Actionability', 'Fail Order Count_curr', '% of Total Failures', '% of Total Orders (Curr)', 'MoM Delta (% of Orders)']].copy()
            display_fail_table.rename(columns={'Psp Operation Status': 'Decline Reason', 'Psp Actionability': 'Actionability', 'Fail Order Count_curr': 'Failures'}, inplace=True)
            
            st.dataframe(
                display_fail_table.style.format({
                    'Failures': "{:,.0f}",
                    '% of Total Failures': "{:.2%}", 
                    '% of Total Orders (Curr)': "{:.2%}", 
                    'MoM Delta (% of Orders)': "{:+.2%}"
                }).map(get_highlighter(0.0001, 0.01, reverse_colors=True), subset=['MoM Delta (% of Orders)']),
                use_container_width=True, hide_index=True, height=400
            )
    else:
        st.info("Insufficient data to calculate failure rates.")


st.header("🏦 Issuer Error Analysis")

if 'issuer_df_memory' not in st.session_state:
    st.session_state.issuer_df_memory = None

# Auto-load the issuer file if it hasn't been loaded yet
if st.session_state.issuer_df_memory is None:
    if os.path.exists(ISSUER_FILE):
        raw_issuer = pd.read_csv(ISSUER_FILE)
        raw_issuer['Date'] = pd.to_datetime(raw_issuer['Date'])
        if 'Card Bin' in raw_issuer.columns:
            raw_issuer['Card Bin'] = raw_issuer['Card Bin'].astype(str).str.replace(r'\.0$', '', regex=True)
        st.session_state.issuer_df_memory = raw_issuer
    else:
        st.warning(f"⚠️ Could not find {ISSUER_FILE} in the repository. Please upload it to GitHub.")

# Pull from memory
issuer_df = st.session_state.issuer_df_memory

if issuer_df is not None:
    # Always operate on a copy of the memory dataframe so filters don't permanently delete rows!
    issuer_df = issuer_df.copy()
    
    # =====================================================================
    # 🔗 APPLYING GLOBAL DASHBOARD FILTERS TO ISSUER DATA
    # =====================================================================
    try:
        # 1. Entity / Data Source Filter
        try:
            if sel_source != 'All' and 'Country' in issuer_df.columns:
                brand_mapping = {
                    'Taiwan': 'Foodpanda', 'Malaysia': 'Foodpanda', 'Hong Kong': 'Foodpanda', 
                    'Singapore': 'Foodpanda', 'Philippines': 'Foodpanda', 'Pakistan': 'Foodpanda', 
                    'Bangladesh': 'Foodpanda', 'Cambodia': 'Foodpanda', 'Myanmar': 'Foodpanda', 'Laos': 'Foodpanda',
                    'Sweden': 'Foodora', 'Norway': 'Foodora', 'Austria': 'Foodora', 
                    'Hungary': 'Foodora', 'Czech Republic': 'Foodora',
                    'Turkey': 'Yemeksepeti'
                }
                issuer_df['data_source'] = issuer_df['Country'].map(brand_mapping).fillna('Pandora')
                issuer_df = issuer_df[issuer_df['data_source'] == sel_source]
        except NameError: 
            pass 
            
        # 2. Country Filter
        try:
            if sel_country != 'All' and 'Country' in issuer_df.columns:
                issuer_df = issuer_df[issuer_df['Country'] == sel_country]
        except NameError: 
            pass 
            
        # 3. Payment Method Filter
        try:
            if sel_pm != 'All' and 'Payment Method' in issuer_df.columns:
                issuer_df = issuer_df[issuer_df['Payment Method'].astype(str).str.lower() == str(sel_pm).lower()]
        except NameError: 
            pass 
            
    except Exception as e:
        st.warning(f"Issuer Filter error: {e}")
    # =====================================================================

    # Identify Current and Previous Months for MoM calculations
    months_available = sorted(issuer_df['Date'].unique())
    if len(months_available) >= 2:
        current_month = months_available[-1]
        prev_month = months_available[-2]
    elif len(months_available) == 1:
        current_month = months_available[0]
        prev_month = None
    else:
        current_month = None
        prev_month = None

    if current_month:
        st.write(f"**Analyzing Data for:** {current_month.strftime('%B %Y')}")
        if prev_month:
            st.caption(f"Comparing MoM against: {prev_month.strftime('%B %Y')}")

        st.markdown("### Filter by PSP & Error Code")
        col_psp, col_err = st.columns(2)
        
        # --- NEW: PSP NAME FILTER ---
        with col_psp:
            if 'Psp Name' in issuer_df.columns:
                psp_options = ['All'] + sorted(issuer_df['Psp Name'].dropna().unique().tolist())
                selected_psp = st.selectbox("⚙️ Select PSP Name:", psp_options)
                
                # Apply the filter if not "All"
                if selected_psp != 'All':
                    issuer_df = issuer_df[issuer_df['Psp Name'] == selected_psp]
            else:
                st.warning("⚠️ 'Psp Name' not found in dataset. Re-run aggregation script.")
                selected_psp = 'All'
                
        # --- ERROR CODE FILTER ---
        with col_err:
            if 'Authorization Status' in issuer_df.columns:
                error_codes = issuer_df[issuer_df['Authorization Status'] != 'Approved']['Acquirer Response'].dropna().unique()
            else:
                error_codes = issuer_df['Acquirer Response'].dropna().unique()
                
            selected_error = st.selectbox("🔍 Select Error Code:", sorted(error_codes))

        if selected_error:
            # --- DATA PROCESSING ---
            # Total global failed transactions for THIS specific error in the current month
            global_error_current = issuer_df[(issuer_df['Date'] == current_month) & (issuer_df['Acquirer Response'] == selected_error)]['Failed Trx Count'].sum()

            # Calculate TOTAL transactions per issuer (Current Month) 
            issuer_total_trx = issuer_df[issuer_df['Date'] == current_month].groupby('Issuer')['Trx Count'].sum().reset_index()
            issuer_total_trx.rename(columns={'Trx Count': 'Total Issuer Trx'}, inplace=True)

            # Calculate ERROR metrics per issuer (Current Month)
            err_curr = issuer_df[(issuer_df['Date'] == current_month) & (issuer_df['Acquirer Response'] == selected_error)]
            issuer_err_curr = err_curr.groupby('Issuer')['Failed Trx Count'].sum().reset_index()
            issuer_err_curr.rename(columns={'Failed Trx Count': 'Error Count'}, inplace=True)

            # Calculate ERROR metrics per issuer (Previous Month) for MoM
            if prev_month:
                err_prev = issuer_df[(issuer_df['Date'] == prev_month) & (issuer_df['Acquirer Response'] == selected_error)]
                issuer_err_prev = err_prev.groupby('Issuer')['Failed Trx Count'].sum().reset_index()
                issuer_err_prev.rename(columns={'Failed Trx Count': 'Prev Error Count'}, inplace=True)
            else:
                issuer_err_prev = pd.DataFrame(columns=['Issuer', 'Prev Error Count'])

            # Merge everything together
            merged_df = pd.merge(issuer_err_curr, issuer_total_trx, on='Issuer', how='left')
            merged_df = pd.merge(merged_df, issuer_err_prev, on='Issuer', how='left').fillna(0)

            # Calculate the required Percentages and MoM
            merged_df['% of Total Trx'] = (merged_df['Error Count'] / merged_df['Total Issuer Trx']) * 100
            merged_df['% of Total Errors'] = np.where(global_error_current > 0, (merged_df['Error Count'] / global_error_current) * 100, 0)
            merged_df['MoM (%)'] = np.where(
                merged_df['Prev Error Count'] > 0,
                ((merged_df['Error Count'] - merged_df['Prev Error Count']) / merged_df['Prev Error Count']) * 100,
                np.nan
            )

            # Sort from largest transaction errors to lowest
            merged_df = merged_df.sort_values(by='Error Count', ascending=False).reset_index(drop=True)

            display_df = merged_df[['Issuer', 'Error Count', '% of Total Trx', '% of Total Errors', 'MoM (%)']].copy()
            
            # Display the Top Level Table
            st.subheader(f"Top Issuers for: '{selected_error}'")
            st.dataframe(
                display_df,
                column_config={
                    "Error Count": st.column_config.NumberColumn("Error Count", format="%d 🛑"),
                    "% of Total Trx": st.column_config.ProgressColumn("% of Total Trx", format="%.2f%%", min_value=0, max_value=100),
                    "% of Total Errors": st.column_config.NumberColumn("% of Total Errors (Share)", format="%.2f%%"),
                    "MoM (%)": st.column_config.NumberColumn("MoM Development", format="%.1f%%")
                },
                hide_index=True,
                use_container_width=True
            )

            # Extend to Card Bin Level (+)
            st.divider()
            st.subheader("💳 Drill-down: Card Bin Level")
            st.markdown("Select an issuer from the list to investigate which specific Card Bins are driving these errors.")
            
            if not merged_df.empty:
                drilldown_issuer = st.selectbox("Select Issuer to Expand:", merged_df['Issuer'].tolist())

                if drilldown_issuer:
                    bin_data = err_curr[err_curr['Issuer'] == drilldown_issuer].groupby('Card Bin')['Failed Trx Count'].sum().reset_index()
                    bin_data = bin_data.sort_values(by='Failed Trx Count', ascending=False).reset_index(drop=True)
                    
                    issuer_total_errors = bin_data['Failed Trx Count'].sum()
                    bin_data['% of Issuer Errors'] = np.where(issuer_total_errors > 0, (bin_data['Failed Trx Count'] / issuer_total_errors) * 100, 0)
                    
                    st.write(f"**Card Bins driving '{selected_error}' for {drilldown_issuer}:**")
                    st.dataframe(
                        bin_data,
                        column_config={
                            "Card Bin": st.column_config.TextColumn("Card Bin (First 6-8 digits)"),
                            "Failed Trx Count": st.column_config.NumberColumn("Error Count", format="%d"),
                            "% of Issuer Errors": st.column_config.ProgressColumn("% Contribution", format="%.1f%%", min_value=0, max_value=100)
                        },
                        hide_index=True
                    )
            else:
                st.info(f"No transactions found for error code '{selected_error}' with the current filters.")


# --- 8. GEMINI AI ---
st.divider()
st.header("🤖 AI Analysis Hub")
api_key = st.text_input("Enter Gemini API Key to activate AI features:", type="password")

tab1, tab2 = st.tabs(["📝 Generate Executive Summary", "💬 Ask Custom Question"])

# Background context for AI
cs_prompt = merged.groupby('Country').agg({'Order Count_curr': 'sum', 'Order Count_prev': 'sum', 'Acceptance Orders_curr': 'sum', 'Acceptance Orders_prev': 'sum', 'Subtotal_Rate_Impact': 'sum', 'Subtotal_Mix_Impact': 'sum'}).reset_index()
cs_prompt['AR_curr'] = np.where(cs_prompt['Order Count_curr']>0, cs_prompt['Acceptance Orders_curr']/cs_prompt['Order Count_curr'], 0)
cs_prompt['AR_prev'] = np.where(cs_prompt['Order Count_prev']>0, cs_prompt['Acceptance Orders_prev']/cs_prompt['Order Count_prev'], 0)
cs_prompt['MoM_Delta'] = cs_prompt['AR_curr'] - cs_prompt['AR_prev']
cs_prompt = cs_prompt.sort_values(by='MoM_Delta', ascending=False)
for col in ['AR_curr', 'MoM_Delta', 'Subtotal_Rate_Impact', 'Subtotal_Mix_Impact']: cs_prompt[col] = cs_prompt[col].apply(lambda x: f"{x:.4%}")

merged['GT_Total_Impact'] = merged['GT_Mix_Impact'] + merged['GT_Rate_Impact']
td_prompt = merged.sort_values(by='GT_Total_Impact', ascending=False).copy()
for col in ['AR_prev', 'AR_curr', 'MoM_Delta', 'GT_Rate_Impact', 'GT_Mix_Impact']: td_prompt[col] = td_prompt[col].apply(lambda x: f"{x:.4%}")

with tab1:
    if st.button("Generate Detailed Summary", key="ai_summary_btn"):
        if not api_key: st.warning("Please enter your Gemini API key above.")
        else:
            with st.spinner("Analyzing..."):
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    prompt = f"""
                    You are a Senior Payments Data Scientist providing an insightful analysis of Month-over-Month (MoM) Acceptance Rate (AR) shifts. 
                    Macro Data: Total AR Change: {tot_del:.4%} (Rate Impact: {tot_perf:.4%}, Mix Impact: {tot_mix:.4%})
                    Country Data:
                    {cs_prompt[['Country', 'AR_curr', 'MoM_Delta', 'Subtotal_Rate_Impact', 'Subtotal_Mix_Impact']].to_string(index=False)}
                    Top Drivers (+):
                    {td_prompt[['Country', 'First Payment Method', 'AR_prev', 'AR_curr', 'MoM_Delta', 'GT_Rate_Impact', 'GT_Mix_Impact']].head(15).to_string(index=False)}
                    Top Drivers (-):
                    {td_prompt[['Country', 'First Payment Method', 'AR_prev', 'AR_curr', 'MoM_Delta', 'GT_Rate_Impact', 'GT_Mix_Impact']].tail(15).to_string(index=False)}
                    
                    Task: Write an executive summary. Follow this EXACT structure:
                    ### 🌍 Country Performance & Anomalies
                    - **Country Summary**: Markdown table (Country | Latest AR | MoM Delta | Rate Impact | Mix Impact). Order by MoM Delta. Wrap ONLY percentages in HTML color tags (`<span style="color:green">`, `<span style="color:red">`).
                    - **Insightful Country Trends**: 3-4 sentences detailing anomalies or counter-balancing shifts.
                    ### 🏢 Global Entity Drivers
                    - **The Heavyweights**: 3-4 sentences identifying specific payment methods moving the global needle. Look for weird volume shifts.
                    ### 🎯 Strategic Recommendations
                    3 specific action items for the payments team based on the data.
                    """
                    response = model.generate_content(prompt)
                    st.markdown(response.text, unsafe_allow_html=True)
                except Exception as e: st.error(f"Error: {e}")

with tab2:
    st.write("Ask a specific question (e.g., 'What drove the drop in generic credit cards in Malaysia? What were the failure reasons?')")
    user_query = st.text_area("Your question:")
    if st.button("Ask AI", key="ai_custom_query"):
        if not api_key: st.warning("Please enter your Gemini API key above.")
        elif not user_query: st.warning("Please enter a question.")
        else:
            with st.spinner("AI is analyzing the full datasets (AR + Failures)..."):
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    underlying_data = df.groupby(['Month', 'Country', 'CC VS APM', 'First Payment Method'])[['Order Count', 'Success Order Count', 'Recovery Order Count', 'Acceptance Orders']].sum().reset_index()
                    chat_master = merged[['Country', 'CC VS APM', 'First Payment Method', 'Order Count_prev', 'Order Count_curr', 'MoM_Delta', 'Subtotal_Rate_Impact', 'Subtotal_Mix_Impact', 'GT_Mix_Impact']].copy()
                    chat_master['Volume_Shift_Country'] = merged['W_sub_curr'] - merged['W_sub_prev']
                    for col in ['MoM_Delta', 'Subtotal_Rate_Impact', 'Subtotal_Mix_Impact', 'GT_Mix_Impact', 'Volume_Shift_Country']: chat_master[col] = chat_master[col].apply(lambda x: f"{x:.4%}")
                    
                    fail_context = "No failure data uploaded."
                    if fail_df is not None:
                        fail_recent = fail_df[fail_df['Month'].isin([curr_month, prev_month])]
                        fail_agg = fail_recent.groupby(['Month', 'Country', 'First Payment Method', 'Psp Operation Status', 'Psp Actionability'])['Fail Order Count'].sum().reset_index()
                        fail_agg = fail_agg[fail_agg['Fail Order Count'] > 0].sort_values(by='Fail Order Count', ascending=False)
                        fail_context = fail_agg.to_csv(index=False)

                    prompt = f"""
                    You are a Senior Payments Data Scientist answering an ad-hoc question.
                    CONTEXT 1: Base Volume (Monthly)
                    {underlying_data.to_csv(index=False)}
                    CONTEXT 2: MoM Variance & Mix Shifts
                    {chat_master.to_csv(index=False)}
                    CONTEXT 3: Top Payment Failure Reasons (Raw Counts by Month)
                    {fail_context}
                    
                    Question: "{user_query}"
                    Instructions:
                    1. Answer directly and concisely based ONLY on the data.
                    2. If asked about performance drops, check CONTEXT 3 to cite the exact failure reasons (Psp Operation Status) causing it.
                    """
                    response = model.generate_content(prompt)
                    st.markdown(response.text)
                except Exception as e: st.error(f"Error: {e}")
