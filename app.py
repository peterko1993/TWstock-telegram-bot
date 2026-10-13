import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import json
import twstock

# 頁面配置
st.set_page_config(
    page_title="台股收益價值投資篩選器",
    page_icon="🏛️",
    layout="wide"
)

def get_stock_name(code: str) -> str:
    """取得台股繁體中文名稱"""
    if code in twstock.codes:
        return twstock.codes[code].name
    return code

def send_to_discord(webhook_url: str, selected_stocks: list):
    """透過 Webhook 推送篩選結果到 Discord 頻道"""
    if not webhook_url or not selected_stocks:
        return False
    
    content = "📢 **【台股收益價值投資雷達 - 最新篩選出爐】**\n"
    content += "────────────────────────\n"
    for item in selected_stocks:
        content += (
            f"📌 **{item['名稱']} ({item['代號']})** | 現價: `{item['現價']:.2f}`\n"
            f"• 營益率: `{item['營益率']:.1f}%` | PE: `{item['PE']:.1f}x` | PBR: `{item['PBR']:.2f}x`\n"
            f"• 近年平均 ROE: `{item['ROE']:.1f}%` | 自主 PEG: `{item['PEG']}`\n"
            f"────────────────────────\n"
        )
    
    payload = {"content": content}
    headers = {"Content-Type": "application/json"}
    response = requests.post(webhook_url, data=json.dumps(payload), headers=headers)
    return response.status_code == 204

@st.cache_data(ttl=3600)
def fetch_stock_fundamental(code: str):
    """抓取單檔股票財報數據並快取 1 小時"""
    try:
        ticker = f"{code}.TW"
        stock = yf.Ticker(ticker)
        df_hist = stock.history(period="5d")
        if df_hist.empty:
            ticker = f"{code}.TWO"
            stock = yf.Ticker(ticker)
            df_hist = stock.history(period="5d")
            if df_hist.empty:
                return None
        
        curr_price = float(df_hist["Close"].iloc[-1])
        fin = stock.financials
        bs = stock.balance_sheet
        info = stock.info or {}

        if fin.empty or bs.empty:
            return None

        # 1. 營業利益率
        op_inc = fin.loc["Operating Income"].iloc[0] if "Operating Income" in fin.index else 0
        rev = fin.loc["Total Revenue"].iloc[0] if "Total Revenue" in fin.index else 1
        op_margin = (op_inc / rev) * 100 if rev > 0 else 0

        # 2. EPS 與 PE
        eps = None
        for eps_key in ["Diluted EPS", "Basic EPS"]:
            if eps_key in fin.index:
                val = fin.loc[eps_key].iloc[0]
                if pd.notna(val) and float(val) > 0:
                    eps = float(val)
                    break
        pe = round(curr_price / eps, 2) if (eps and eps > 0) else None

        # 3. PBR
        pbr = info.get("priceToBook")
        if not pbr or pd.isna(pbr):
            equity = bs.loc["Stockholders Equity"].iloc[0] if "Stockholders Equity" in bs.index else (
                bs.loc["Total Equity Gross Minority Interest"].iloc[0] if "Total Equity Gross Minority Interest" in bs.index else None
            )
            shares = stock.fast_info.shares if hasattr(stock, "fast_info") and hasattr(stock.fast_info, "shares") else None
            if equity and shares and shares > 0:
                bps = equity / shares
                pbr = round(curr_price / bps, 2) if bps > 0 else None
            else:
                pbr = 1.0

        # 4. ROE
        net_incomes = fin.loc["Net Income Common Stockholders"] if "Net Income Common Stockholders" in fin.index else fin.loc["Net Income"]
        equity_series = bs.loc["Stockholders Equity"] if "Stockholders Equity" in bs.index else bs.loc["Total Equity Gross Minority Interest"]
        common_years = net_incomes.index.intersection(equity_series.index)
        roe_series = (net_incomes[common_years] / equity_series[common_years]) * 100
        avg_roe = float(roe_series.mean()) if not roe_series.empty else 0.0

        # 5. 自主精算 PEG
        n_years = len(fin.columns) - 1
        growth_rate = 0.0
        if n_years > 0 and len(net_incomes) > 1:
            ni_latest = float(net_incomes.iloc[0])
            ni_oldest = float(net_incomes.iloc[-1])
            if ni_latest > 0 and ni_oldest > 0:
                growth_rate = (((ni_latest / ni_oldest) ** (1 / n_years)) - 1) * 100

        if pe and pe > 0 and growth_rate > 0:
            peg_val = round(pe / growth_rate, 2)
            peg_str = f"{peg_val:.2f}"
        elif growth_rate <= 0:
            peg_str = "衰退無效"
            peg_val = 999.0
        else:
            peg_str = "無效"
            peg_val = 999.0

        return {
            "代號": code,
            "名稱": get_stock_name(code),
            "現價": round(curr_price, 2),
            "營益率": round(op_margin, 1),
            "PE": round(pe, 1) if pe else None,
            "PBR": round(pbr, 2) if pbr else None,
            "ROE": round(avg_roe, 1),
            "PEG": peg_str,
            "_peg_num": peg_val
        }
    except Exception:
        return None

# ================= UI 介面佈局 =================
st.title("🏛️ 台股收益價值投資篩選雷達")
st.markdown("基於 **Kabu1000 收益價值投資模型**：聚焦營業利益率、本益比、淨值比、ROE 與自主精算 PEG。")

# 側邊欄控制台
with st.sidebar:
    st.header("⚙️ 篩選參數設置")
    min_op_margin = st.slider("最低營業利益率 (%)", 0.0, 30.0, 10.0, step=0.5)
    max_pe = st.slider("最高本益比 (PE)", 5.0, 30.0, 15.0, step=0.5)
    max_pbr = st.slider("最高股價淨值比 (PBR)", 0.5, 3.5, 1.5, step=0.1)
    min_roe = st.slider("最低平均 ROE (%)", 5.0, 25.0, 12.0, step=0.5)
    filter_peg = st.checkbox("限制 PEG ≤ 1.2 (排除估值過高與負成長)", value=False)
    
    st.divider()
    st.header("🔔 Discord 聯動設置")
    webhook_url = st.text_input("Discord Webhook URL", type="password", placeholder="https://discord.com/api/webhooks/...")
    
    st.divider()
    st.header("📋 掃描股票池")
    default_pool = "2330, 2454, 2308, 2379, 3034, 3005, 3653, 6274, 5269, 3529, 6488, 3131, 3293, 2059, 2317, 2382, 3231, 2357, 2449, 3711"
    stock_input = st.text_area("輸入要掃描的台股代碼（逗號分隔）", default_pool, height=120)

stock_list = [s.strip() for s in stock_input.split(",") if s.strip().isdigit()]

# 主面板操作
col1, col2 = st.columns([1, 4])
with col1:
    start_scan = st.button("🚀 開始全速篩選", type="primary", use_container_width=True)

if start_scan:
    progress_bar = st.progress(0)
    status_text = st.empty()
    all_data = []

    for idx, code in enumerate(stock_list):
        status_text.text(f"正在分析 {code} ({idx+1}/{len(stock_list)})...")
        data = fetch_stock_fundamental(code)
        if data:
            all_data.append(data)
        progress_bar.progress((idx + 1) / len(stock_list))

    status_text.empty()
    progress_bar.empty()

    if all_data:
        df = pd.DataFrame(all_data)
        # 依條件過濾
        cond = (
            (df["營益率"] >= min_op_margin) &
            (df["PE"].notna()) & (df["PE"] <= max_pe) &
            (df["PBR"].notna()) & (df["PBR"] <= max_pbr) &
            (df["ROE"] >= min_roe)
        )
        if filter_peg:
            cond = cond & (df["_peg_num"] <= 1.2)

        matched_df = df[cond].copy()

        st.subheader(f"🎯 符合篩選條件標的（共 {len(matched_df)} 檔）")
        if not matched_df.empty:
            display_cols = ["代號", "名稱", "現價", "營益率", "PE", "PBR", "ROE", "PEG"]
            st.dataframe(matched_df[display_cols], use_container_width=True)

            if webhook_url:
                if st.button("📢 將結果發送至 Discord 頻道", use_container_width=False):
                    success = send_to_discord(webhook_url, matched_df.to_dict(orient="records"))
                    if success:
                        st.success("✅ 已成功推送至 Discord！")
                    else:
                        st.error("❌ 推送失敗，請確認 Webhook URL 是否正確。")
            else:
                st.info("💡 提示：在左側側邊欄填入 Discord Webhook URL，即可一鍵將精選清單推送至聊天頻道。")
        else:
            st.warning("⚠️ 目前條件下沒有符合的股票，請嘗試在側邊欄放寬本益比或淨值比門檻。")
    else:
        st.error("未能成功取得任何股票資料，請檢查代碼是否正確。")
