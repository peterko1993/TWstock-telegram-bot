import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import json
import twstock

st.set_page_config(
    page_title="全台股收益價值投資篩選雷達",
    page_icon="🏛️",
    layout="wide"
)

# ================= 官方全市場快速抓取 =================
@st.cache_data(ttl=14400) # 快取 4 小時
def fetch_twse_market_pe_pbr():
    """從證交所 OpenAPI 2 秒下載全上市公司 (約 1000 檔) 本益比與淨值比總表"""
    url = "https://openapi.twse.com.tw/v1/exchangeReport/BWIBBU_d"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            data = res.json()
            stocks = []
            for item in data:
                code = item.get("Code", "").strip()
                name = item.get("Name", "").strip()
                pe_raw = item.get("PEratio", "").replace(",", "")
                pb_raw = item.get("PBratio", "").replace(",", "")
                
                # 只保留 4 位數普通股
                if len(code) == 4 and code.isdigit():
                    try:
                        pe = float(pe_raw) if pe_raw and pe_raw != "-" else None
                        pb = float(pb_raw) if pb_raw and pb_raw != "-" else None
                        stocks.append({
                            "code": code,
                            "name": name,
                            "pe_market": pe,
                            "pb_market": pb,
                            "market": "TW"
                        })
                    except ValueError:
                        continue
            return pd.DataFrame(stocks)
    except Exception as e:
        st.error(f"連線證交所開放資料異常: {e}")
    return pd.DataFrame()

@st.cache_data(ttl=14400)
def fetch_tpex_market_pe_pbr():
    """從櫃買中心開放資料下載全上櫃公司 (約 800 檔) 本益比與淨值比總表"""
    url = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_peratio_analysis"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            data = res.json()
            stocks = []
            for item in data:
                # 兼容不同欄位命名
                code = str(item.get("SecuritiesCompanyCode") or item.get("CompanyCode") or item.get("Code") or "").strip()
                name = str(item.get("CompanyName") or item.get("Name") or "").strip()
                pe_raw = str(item.get("PriceEarningRatio") or item.get("PEratio") or "").replace(",", "")
                pb_raw = str(item.get("PriceBookRatio") or item.get("PBratio") or "").replace(",", "")
                
                if len(code) == 4 and code.isdigit():
                    try:
                        pe = float(pe_raw) if pe_raw and pe_raw != "-" else None
                        pb = float(pb_raw) if pb_raw and pb_raw != "-" else None
                        stocks.append({
                            "code": code,
                            "name": name,
                            "pe_market": pe,
                            "pb_market": pb,
                            "market": "TWO"
                        })
                    except ValueError:
                        continue
            return pd.DataFrame(stocks)
    except Exception:
        pass
    return pd.DataFrame()

# ================= 單檔深入財報精算 =================
@st.cache_data(ttl=3600)
def deep_analyze_stock(code, name, market_type):
    """只對初篩通過的少數股票進行 yfinance 深度分析"""
    try:
        ticker = f"{code}.{market_type}"
        stock = yf.Ticker(ticker)
        df_hist = stock.history(period="5d")
        if df_hist.empty:
            return None
        
        curr_price = float(df_hist["Close"].iloc[-1])
        fin = stock.financials
        bs = stock.balance_sheet
        
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
        equity = bs.loc["Stockholders Equity"].iloc[0] if "Stockholders Equity" in bs.index else (
            bs.loc["Total Equity Gross Minority Interest"].iloc[0] if "Total Equity Gross Minority Interest" in bs.index else None
        )
        shares = stock.fast_info.shares if hasattr(stock, "fast_info") and hasattr(stock.fast_info, "shares") else None
        if equity and shares and shares > 0:
            pbr = round(curr_price / (equity / shares), 2)
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
        else:
            peg_str = "衰退/無效"
            peg_val = 999.0

        return {
            "代號": code,
            "名稱": name,
            "現價": round(curr_price, 2),
            "營益率(%)": round(op_margin, 1),
            "本益比(PE)": round(pe, 1) if pe else None,
            "淨值比(PBR)": round(pbr, 2) if pbr else None,
            "五年均ROE(%)": round(avg_roe, 1),
            "自主PEG": peg_str,
            "_peg_num": peg_val
        }
    except Exception:
        return None

def send_to_discord(webhook_url: str, selected_stocks: list):
    """發布到 Discord 頻道"""
    content = "📢 **【全台股收益價值雷達 - 掃描精選結果】**\n────────────────────────\n"
    for item in selected_stocks:
        content += (
            f"📌 **{item['名稱']} ({item['代號']})** | 現價: `{item['現價']:.2f}`\n"
            f"• 營益率: `{item['營益率(%)']:.1f}%` | PE: `{item['本益比(PE)']}x` | PBR: `{item['淨值比(PBR)']}x`\n"
            f"• 5年均ROE: `{item['五年均ROE(%)']:.1f}%` | 自主PEG: `{item['自主PEG']}`\n────────────────────────\n"
        )
    return requests.post(webhook_url, json={"content": content}).status_code == 204

# ================= UI 介面 =================
st.title("🏛️ 全台股收益價值投資自動篩選雷達")
st.caption("官方 OpenAPI 毫秒級全市場粗篩 ➔ yfinance 深度基本面精算 ➔ Discord 一鍵發布")

with st.sidebar:
    st.header("🔍 掃描市場範圍")
    market_choice = st.radio("選擇掃描市場", ["全台股 (上市+上櫃 約1,800檔)", "僅上市股票 (TWSE 約1,000檔)", "僅上櫃股票 (TPEx 約800檔)"])
    
    st.divider()
    st.header("⚙️ 收益價值門檻設定")
    max_pe = st.slider("第一階段：最高本益比 (PE)", 8.0, 20.0, 15.0, step=0.5)
    max_pbr = st.slider("第一階段：最高淨值比 (PBR)", 0.8, 2.5, 1.5, step=0.1)
    min_op_margin = st.slider("第二階段：最低營益率 (%)", 5.0, 25.0, 10.0, step=0.5)
    min_roe = st.slider("第二階段：最低平均 ROE (%)", 8.0, 20.0, 12.0, step=0.5)
    filter_peg = st.checkbox("嚴格限制 PEG ≤ 1.2 (排除負成長與高估)", value=False)
    
    st.divider()
    webhook_url = st.text_input("Discord Webhook URL", type="password")

if st.button("🚀 開始全市場智慧掃描", type="primary", use_container_width=True):
    # 步驟一：下載官方全市場總表
    with st.spinner("⚡ [1/2] 正在向證交所與櫃買中心下載全市場官方總表..."):
        if "上市+上櫃" in market_choice:
            df_twse = fetch_twse_market_pe_pbr()
            df_tpex = fetch_tpex_market_pe_pbr()
            df_market = pd.concat([df_twse, df_tpex], ignore_index=True)
        elif "僅上市" in market_choice:
            df_market = fetch_twse_market_pe_pbr()
        else:
            df_market = fetch_tpex_market_pe_pbr()
            
    if df_market.empty:
        st.error("無法取得全市場開放資料，請稍候再試。")
    else:
        total_count = len(df_market)
        # 步驟二：第一階段極速過濾
        df_filtered = df_market[
            (df_market["pe_market"].notna()) &
            (df_market["pe_market"] > 0) &
            (df_market["pe_market"] <= max_pe) &
            (df_market["pb_market"].notna()) &
            (df_market["pb_market"] <= max_pbr)
        ].copy()
        
        candidate_count = len(df_filtered)
        st.info(f"📊 全市場共掃描 **{total_count}** 檔股票 ➔ 經初篩 (PE≤{max_pe}, PBR≤{max_pbr}) 後鎖定 **{candidate_count}** 檔潛力股進入深度財報精算！")
        
        # 步驟三：深度財報精算
        progress = st.progress(0)
        status = st.empty()
        deep_results = []
        
        candidates = df_filtered.to_dict(orient="records")
        for i, row in enumerate(candidates):
            status.text(f"正在精算財報: {row['name']} ({row['code']}) [{i+1}/{candidate_count}]...")
            res = deep_analyze_stock(row["code"], row["name"], row["market"])
            if res:
                # 第二階段門檻過濾
                if res["營益率(%)"] >= min_op_margin and res["五年均ROE(%)"] >= min_roe:
                    if not filter_peg or (filter_peg and res["_peg_num"] <= 1.2):
                        deep_results.append(res)
            progress.progress((i + 1) / candidate_count)
            
        progress.empty()
        status.empty()
        
        if deep_results:
            result_df = pd.DataFrame(deep_results)
            st.success(f"🎉 掃描完成！全市場最終挑選出 **{len(result_df)}** 檔符合【收益價值投資】的高勝率卓越標的：")
            cols = ["代號", "名稱", "現價", "營益率(%)", "本益比(PE)", "淨值比(PBR)", "五年均ROE(%)", "自主PEG"]
            st.dataframe(result_df[cols], use_container_width=True)
            
            if webhook_url:
                if st.button("📢 將此精選名單推送至 Discord 頻道"):
                    if send_to_discord(webhook_url, deep_results):
                        st.toast("✅ 已成功推送至 Discord！")
                    else:
                        st.error("推送失敗，請檢查 Webhook 網址。")
        else:
            st.warning("初篩出的候選股在深度財報檢驗中未達標，建議在左側放寬營益率或 ROE 門檻。")
