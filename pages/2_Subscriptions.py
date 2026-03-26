import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import os
import plotly.express as px  # <--- ADD THIS LINE HERE

# --- FILE NAMES ---
# Make sure this matches the exact name of the file you uploaded to your GitHub repo!
SUBS_FILE = "aggregated_subscription_data.csv"

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="Subscription AR", layout="wide", page_icon="🔁")
st.title("🔁 Subscription Acceptance Rates")
st.markdown("*(Analyze Subscription Payments natively split by **CIT** and **MIT**)*")

# --- 2. REUSABLE FUNCTIONS ---
def get_highlighter(threshold, max_scale, reverse_colors=False):
    def highlight(val):
        if not isinstance(val, (int, float)) or pd.isna(val): return ''
        
        # Determine colors based on whether we want "Higher is Better" or "Higher is Worse"
        # Green: 39, 174, 96 | Red: 231, 76, 60
        positive_color = '231, 76, 60' if reverse_colors else '39, 174, 96'
        negative_color = '39, 174, 96' if reverse_colors else '231, 76, 60'
        
        if val >= threshold:
            intensity = min(val / max_scale, 1) 
            return f'background-color: rgba({positive_color}, {0.15 + (0.4 * intensity)});'
        elif val <= -threshold:
            intensity = min(abs(val) / max_scale, 1)
            return f'background-color: rgba({negative_color}, {0.15 + (0.4 * intensity)});'
        return ''
    return highlight

def calculate_variance(curr, prev, join_keys):
    merged = pd.merge(curr, prev, on=join_keys, how='outer', suffixes=('_curr', '_prev')).fillna(0)
    
    orders_curr = merged['Order Count_curr'].sum()
    orders_prev = merged['Order Count_prev'].sum()
    
    ar_curr = merged['Acceptance Orders_curr'].sum() / orders_curr if orders_curr > 0 else 0
    ar_prev = merged['Acceptance Orders_prev'].sum() / orders_prev if orders_prev > 0 else 0
    global_delta = ar_curr - ar_prev
    
    merged['AR_curr'] = np.where(merged['Order Count_curr']>0, merged['Acceptance Orders_curr']/merged['Order Count_curr'], 0)
    merged['AR_prev'] = np.where(merged['Order Count_prev']>0, merged['Acceptance Orders_prev']/merged['Order Count_prev'], 0)
    
    merged['Weight_curr'] = merged['Order Count_curr'] / orders_curr if orders_curr > 0 else 0
    merged['Weight_prev'] = merged['Order Count_prev'] / orders_prev if orders_prev > 0 else 0
    
    merged['Rate_Impact'] = (merged['AR_curr'] - merged['AR_prev']) * merged['Weight_prev']
    merged['Mix_Impact'] = (merged['Weight_curr'] - merged['Weight_prev']) * (merged['AR_curr'] - ar_prev)
    
    global_perf = merged['Rate_Impact'].sum()
    global_mix = merged['Mix_Impact'].sum()
    
    return merged, global_delta, global_perf, global_mix

# --- 3. DATA LOADING (AUTOMATED FROM GITHUB REPO) ---
st.sidebar.markdown("### 📥 Data Source")

# 1. Create the memory bank if it doesn't exist yet
if 'subs_df' not in st.session_state:
    st.session_state.subs_df = None

# 2. If memory is empty, auto-load the file from the local repository directory
if st.session_state.subs_df is None:
    if os.path.exists(SUBS_FILE):
        with st.spinner(f"Loading {SUBS_FILE}..."):
            try:
                is_csv = SUBS_FILE.endswith('.csv') or SUBS_FILE.endswith('.gz')
                if is_csv:
                    # We try tab-separated first, then comma
                    temp_df = pd.read_csv(SUBS_FILE, sep='\t')
                    if len(temp_df.columns) < 5:
                        temp_df = pd.read_csv(SUBS_FILE)
                    df = temp_df
                else:
                    df = pd.read_excel(SUBS_FILE)
                    
                rename_map = {
                    'Subscr Date': 'Order Date', 'platform': 'data_source',
                    'CC or APM': 'CC VS APM', 'Simple PM': 'First Payment Method',
                    'Subscriptions Count': 'Order Count', 'Success Subscr Count': 'Success Order Count',
                    'Recovered Subscr Count': 'Recovery Order Count', 'Fail Subscr Count': 'Fail Order Count',
                    'Decline Reasons': 'Fail Reason'
                }
                df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
                df = df.fillna(0)
                
                if 'Month' not in df.columns:
                    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
                    df['Month'] = df['Order Date'].dt.to_period('M').astype(str)
                
                for col in ['Order Count', 'Success Order Count', 'Recovery Order Count', 'Fail Order Count']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    
                df['Acceptance Orders'] = df.get('Success Order Count', 0) + df.get('Recovery Order Count', 0)
                
                if 'Tx_Type' not in df.columns:
                    if 'CIT/MIT' in df.columns:
                        df['Tx_Type'] = np.where(df['CIT/MIT'].astype(str).str.upper().str.contains('MIT'), 'MIT', 'CIT')
                    elif 'cit_mit' in df.columns:
                        df['Tx_Type'] = np.where(df['cit_mit'].astype(str).str.upper().str.contains('MIT'), 'MIT', 'CIT')
                    else:
                        df['Tx_Type'] = 'UNKNOWN'
                    
                brand_mapping = {
                    'Taiwan': 'Foodpanda', 'Malaysia': 'Foodpanda', 'Hong Kong': 'Foodpanda', 
                    'Singapore': 'Foodpanda', 'Philippines': 'Foodpanda', 'Pakistan': 'Foodpanda', 
                    'Bangladesh': 'Foodpanda', 'Cambodia': 'Foodpanda', 'Myanmar': 'Foodpanda', 'Laos': 'Foodpanda',
                    'Sweden': 'Foodora', 'Norway': 'Foodora', 'Austria': 'Foodora', 
                    'Turkey': 'Yemeksepeti'
                }
                if 'data_source' in df.columns and 'Country' in df.columns:
                    is_pandora = df['data_source'].astype(str).str.strip() == 'Pandora'
                    df.loc[is_pandora, 'data_source'] = df.loc[is_pandora, 'Country'].map(brand_mapping).fillna('Pandora')

                # SAVE TO LONG-TERM MEMORY
                st.session_state.subs_df = df

            except Exception as e:
                st.error(f"🚨 Data Prep Error: {e}")
                st.stop()
    else:
        st.warning(f"⚠️ Could not find '{SUBS_FILE}' in the repository. Please upload it to GitHub.")

# 3. Use the memory!
if st.session_state.subs_df is not None:
    st.sidebar.success("✅ Subscription Data securely loaded from repository.")
    df = st.session_state.subs_df.copy()
else:
    st.stop()
    
# --- 4. CIT/MIT HARD TOGGLE ---
st.divider()
st.markdown("### 🎛️ Select Transaction Context")
st.markdown("*(All charts and metrics below will strictly filter to only show your selection)*")
tx_view = st.radio("Isolate Data By:", ["CIT (Customer Initiated)", "MIT (Merchant Initiated)"], horizontal=True, label_visibility="collapsed")

# Apply the hard filter to the entire dataframe before any math happens!
selected_tx = 'MIT' if 'MIT' in tx_view else 'CIT'
df = df[df['Tx_Type'] == selected_tx]

# --- 5. MONTHLY CALCULATIONS ---
months = sorted(df['Month'].unique())
if len(months) < 2:
    st.info("Please ensure data has at least two distinct months to view MoM attribution.")
    st.stop()

curr_month, prev_month = months[-1], months[-2]

# --- 6. CASCADING FILTERS (Standard Dropdown Style) ---
st.divider()
st.sidebar.markdown(f"### 🎯 Filters ({selected_tx} Context)")

# 1. Entity Filter
entities = ["All"] + sorted([str(e) for e in df['data_source'].unique() if str(e) not in ['0', 'Unknown']])
# We use selectbox (Standard dropdown) instead of multiselect (Bubbles/Chips)
selected_entity = st.sidebar.selectbox("Filter by Entity", options=entities, index=0)

df_filtered = df.copy()
if selected_entity != "All":
    df_filtered = df_filtered[df_filtered['data_source'] == selected_entity]

# 2. Country Filter
countries = ["All"] + sorted([str(c) for c in df_filtered['Country'].unique() if str(c) not in ['0', 'Unknown']])
selected_country = st.sidebar.selectbox("Filter by Country", options=countries, index=0)

if selected_country != "All":
    df_filtered = df_filtered[df_filtered['Country'] == selected_country]

# 3. CC vs APM Filter
if 'CC VS APM' in df_filtered.columns:
    cc_apm_types = ["All"] + sorted([str(c) for c in df_filtered['CC VS APM'].unique() if str(c) not in ['0', 'Unknown']])
    selected_cc_apm = st.sidebar.selectbox("Filter by CC vs APM", options=cc_apm_types, index=0)
    if selected_cc_apm != "All":
        df_filtered = df_filtered[df_filtered['CC VS APM'] == selected_cc_apm]

# 4. Payment Method Filter
if 'First Payment Method' in df_filtered.columns:
    payment_methods = ["All"] + sorted([str(p) for p in df_filtered['First Payment Method'].unique() if str(p) not in ['0', 'Unknown']])
    selected_pm = st.sidebar.selectbox("Filter by Payment Method", options=payment_methods, index=0)
    if selected_pm != "All":
        df_filtered = df_filtered[df_filtered['First Payment Method'] == selected_pm]

# Final Data Split for calculations
df_month_curr = df_filtered[df_filtered['Month'] == curr_month]
df_month_prev = df_filtered[df_filtered['Month'] == prev_month]

# --- 7. GLOBAL PERFORMANCE HEADER ---
st.header(f"🌍 {selected_tx} Performance: {curr_month} vs {prev_month}")

agg_cols = ['Order Count', 'Acceptance Orders']
join_keys = ['data_source', 'CC VS APM']

df_global_curr = df_month_curr.groupby(join_keys)[agg_cols].sum().reset_index()
df_global_prev = df_month_prev.groupby(join_keys)[agg_cols].sum().reset_index()

global_merged, g_delta, g_perf, g_mix = calculate_variance(df_global_curr, df_global_prev, join_keys)
global_ar = df_global_curr['Acceptance Orders'].sum() / df_global_curr['Order Count'].sum() if df_global_curr['Order Count'].sum() > 0 else 0

col1, col2, col3 = st.columns(3)
col1.metric(f"Global {selected_tx} AR (Latest)", f"{global_ar:.2%}", f"{g_delta:+.2%}")
col2.metric("Global Performance Driver", f"{g_perf:+.2%}")
col3.metric("Global Mix Driver", f"{g_mix:+.2%}")

# --- 8. ENTITY ROLLUP TABLE ---
st.markdown(f"#### 🏢 Brand Breakdown ({selected_tx})")
entity_summary = global_merged.groupby(['data_source']).agg({
    'Order Count_curr': 'sum', 'Order Count_prev': 'sum',
    'Acceptance Orders_curr': 'sum', 'Acceptance Orders_prev': 'sum',
    'Rate_Impact': 'sum', 'Mix_Impact': 'sum'
}).reset_index()

entity_summary['Latest AR'] = np.where(entity_summary['Order Count_curr']>0, entity_summary['Acceptance Orders_curr']/entity_summary['Order Count_curr'], 0)
entity_summary['Prev AR'] = np.where(entity_summary['Order Count_prev']>0, entity_summary['Acceptance Orders_prev']/entity_summary['Order Count_prev'], 0)
entity_summary['MoM Delta'] = entity_summary['Latest AR'] - entity_summary['Prev AR']

display_cols = ['data_source', 'Latest AR', 'MoM Delta', 'Rate_Impact', 'Mix_Impact']
global_display = entity_summary[display_cols].sort_values(by='data_source').rename(
    columns={'data_source': 'Entity', 'Rate_Impact': 'Global Rate Impact', 'Mix_Impact': 'Global Mix Impact'}
)

format_cols = ['Latest AR', 'MoM Delta', 'Global Rate Impact', 'Global Mix Impact']
st.dataframe(
    global_display.style.format({c: "{:.2%}" for c in format_cols})
    .map(get_highlighter(0.001, 0.05), subset=['MoM Delta', 'Global Rate Impact', 'Global Mix Impact']),
    use_container_width=True, hide_index=True
)

# --- 9. TOP DECLINE REASONS (CORE BUSINESS STYLE) ---
st.divider()
st.header(f"🛑 {selected_tx} Decline Drivers ({curr_month} vs {prev_month})")

# 1. Get total orders (the denominator) for the filtered context
total_orders_curr = df_month_curr['Order Count'].sum()
total_orders_prev = df_month_prev['Order Count'].sum()

# 2. Extract Failures (Only rows with Fail Order Count > 0)
fail_curr = df_month_curr[df_month_curr['Fail Order Count'] > 0].groupby('Fail Reason')['Fail Order Count'].sum().reset_index()
fail_prev = df_month_prev[df_month_prev['Fail Order Count'] > 0].groupby('Fail Reason')['Fail Order Count'].sum().reset_index()

if total_orders_curr > 0:
    # 3. Merge current and previous for comparison
    fail_merged = pd.merge(fail_curr, fail_prev, on='Fail Reason', suffixes=('_curr', '_prev'), how='outer').fillna(0)
    
    # 4. Calculate Rates and MoM Delta
    fail_merged['% of Total Orders (Curr)'] = fail_merged['Fail Order Count_curr'] / total_orders_curr
    fail_merged['% of Total Orders (Prev)'] = fail_merged['Fail Order Count_prev'] / total_orders_prev if total_orders_prev > 0 else 0
    fail_merged['MoM Delta (% of Orders)'] = fail_merged['% of Total Orders (Curr)'] - fail_merged['% of Total Orders (Prev)']
    
    # 5. Simple Actionability Mapping for Subscriptions
    # Insufficient funds and Generic rejections are usually non-actionable; 3DS/Authentication are actionable.
    def map_actionability(reason):
        reason = str(reason).lower()
        if any(x in reason for x in ['3d', 'authentication', 'invalid merchant', 'expired']): return 'Actionable'
        return 'Non-actionable'
    
    fail_merged['Actionability'] = fail_merged['Fail Reason'].apply(map_actionability)
    
    # 6. Prep Display
    total_fails_curr = fail_merged['Fail Order Count_curr'].sum()
    fail_merged['% of Total Failures'] = fail_merged['Fail Order Count_curr'] / total_fails_curr if total_fails_curr > 0 else 0
    
    # Sort by the most impactful failures
    fail_display = fail_merged[fail_merged['Fail Order Count_curr'] > 0].sort_values(by='% of Total Orders (Curr)', ascending=False).head(20)

    # 7. Layout: Chart on Left, Table on Right
    col_chart, col_table = st.columns([1, 1.5])

    with col_chart:
        fig_fail = px.bar(
            fail_display.head(10).sort_values(by='% of Total Orders (Curr)', ascending=True), 
            x='% of Total Orders (Curr)', 
            y='Fail Reason', 
            color='Actionability', 
            orientation='h',
            text='% of Total Orders (Curr)',
            color_discrete_map={'Actionable': '#f39c12', 'Non-actionable': '#e74c3c'},
            title=f"Top {selected_tx} Reasons (% of Total Orders)"
        )
        fig_fail.update_traces(texttemplate='%{text:.2%}', textposition='outside')
        fig_fail.update_layout(xaxis_tickformat='.1%', showlegend=True, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_fail, use_container_width=True)

    with col_table:
        # Style the table exactly like Core Business (Reverse colors: Positive delta = Red)
        display_fail_table = fail_display[['Fail Reason', 'Actionability', 'Fail Order Count_curr', '% of Total Failures', '% of Total Orders (Curr)', 'MoM Delta (% of Orders)']].copy()
        display_fail_table.rename(columns={'Fail Order Count_curr': 'Failures'}, inplace=True)
        
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
    st.info("Insufficient data to calculate failure rates for this selection.")

# --- 10. AI ANALYSIS HUB ---
st.divider()
st.header("🤖 AI Analysis Hub (Subscriptions)")
api_key = st.text_input("Enter Gemini API Key to activate AI features:", type="password")

tab1, tab2 = st.tabs(["📝 Generate Executive Summary", "💬 Ask Custom Question"])

# --- PREPARE DATA FOR AI ---
# 1. Country Summary for AI
agg_cols_ai = ['Order Count', 'Acceptance Orders']
cs_curr = df_month_curr.groupby('Country')[agg_cols_ai].sum().reset_index()
cs_prev = df_month_prev.groupby('Country')[agg_cols_ai].sum().reset_index()
cs_merged, _, _, _ = calculate_variance(cs_curr, cs_prev, 'Country')

cs_prompt = cs_merged[['Country', 'AR_curr', 'AR_prev', 'Rate_Impact', 'Mix_Impact']].copy()
for col in ['AR_curr', 'AR_prev', 'Rate_Impact', 'Mix_Impact']:
    cs_prompt[col] = cs_prompt[col].apply(lambda x: f"{x:.2%}")

# 2. Detailed Drivers for AI (Country + Payment Method)
td_curr = df_month_curr.groupby(['Country', 'First Payment Method'])[agg_cols_ai].sum().reset_index()
td_prev = df_month_prev.groupby(['Country', 'First Payment Method'])[agg_cols_ai].sum().reset_index()
td_merged, _, _, _ = calculate_variance(td_curr, td_prev, ['Country', 'First Payment Method'])
td_merged['Total_Impact'] = td_merged['Rate_Impact'] + td_merged['Mix_Impact']
td_prompt = td_merged.sort_values(by='Total_Impact', ascending=False).copy()

for col in ['AR_curr', 'AR_prev', 'Rate_Impact', 'Mix_Impact']:
    td_prompt[col] = td_prompt[col].apply(lambda x: f"{x:.2%}")

# 3. Failure Context
fail_context_df = df_month_curr[df_month_curr['Fail Order Count'] > 0].groupby('Fail Reason')['Fail Order Count'].sum().reset_index()
fail_context_df = fail_context_df.sort_values(by='Fail Order Count', ascending=False).head(10)

with tab1:
    if st.button("Generate Subscription Summary", key="ai_sub_summary"):
        if not api_key:
            st.warning("Please enter your Gemini API key above.")
        else:
            with st.spinner("Analyzing Subscription Trends..."):
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-2.0-flash') # or your preferred version
                    
                    prompt = f"""
                    You are a Senior Payments Data Scientist. Analyze this Subscription AR data ({selected_tx} transactions).
                    
                    MACRO STATS ({curr_month} vs {prev_month}):
                    - Total AR Change: {g_delta:+.2%}
                    - Performance Driver: {g_perf:+.2%}
                    - Mix Driver: {g_mix:+.2%}
                    
                    COUNTRY BREAKDOWN:
                    {cs_prompt.to_string(index=False)}
                    
                    TOP PAYMENT METHOD DRIVERS:
                    {td_prompt[['Country', 'First Payment Method', 'AR_prev', 'AR_curr', 'Rate_Impact', 'Mix_Impact']].head(10).to_string(index=False)}
                    
                    TOP FAILURE REASONS:
                    {fail_context_df.to_string(index=False)}

                    TASK: Write an executive summary focusing on why {selected_tx} AR moved.
                    Structure:
                    ### 📊 Performance Overview ({selected_tx})
                    (One paragraph summarizing the 'Why' behind the global move)
                    
                    ### 🏳️ Top Country Movers
                    (Markdown table of top 3 winners and top 3 losers with 1 sentence analysis)
                    
                    ### 🚨 Failure Analysis
                    (Analyze if specific failure reasons are spiking and their impact)
                    
                    ### 💡 Action Items
                    (3 specific tactical recommendations)
                    """
                    response = model.generate_content(prompt)
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"AI Error: {e}")

with tab2:
    user_query = st.text_area("Ask a specific question about these subscriptions:")
    if st.button("Ask AI", key="ai_sub_query"):
        if not api_key:
            st.warning("Please enter your Gemini API key.")
        elif not user_query:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Scanning data..."):
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-2.0-flash')
                    
                    # Provide raw context for the custom question
                    context = f"""
                    Filtered Data Context ({selected_tx}):
                    Detailed Variance: {td_merged.head(50).to_csv(index=False)}
                    Failures: {fail_context_df.to_csv(index=False)}
                    """
                    
                    full_prompt = f"Data Context:\n{context}\n\nQuestion: {user_query}"
                    response = model.generate_content(full_prompt)
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"AI Error: {e}")
