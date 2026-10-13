import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import json
import twstock
import time
from collections import defaultdict

st.set_page_config(
    page_title="全台股收益價值投資篩選雷達",
    page_icon="🏛️",
    layout="wide"
)

# ================= 1. 安全讀取 Secrets =================
DEFAULT_WEBHOOK = ""
try:
    if hasattr(st, "secrets") and "DISCORD_WEBHOOK_URL" in st.secrets:
        DEFAULT_WEBHOOK = str(st.secrets["DISCORD_WEBHOOK_URL"]).strip()
except Exception:
    pass

# ================= 2. 狀態常駐 =================
if "scan_results" not in st.session_state:
    st.session_state["scan_results"] = None
if "candidate_count" not in st.session_state:
    st.session_state["candidate_count"] = 0
if "total_count" not in st.session_state:
    st.session_state["total_count"] = 0

def get_stock_industry(code: str) -> str:
    """取得台股官方標準產業分類"""
    if code in twstock.codes:
        group = twstock.codes[code].group
        if group and group.strip():
            return group.strip()
    return "其他類"

# ================= 官方全市場快速抓取 =================
@st.cache_data(ttl=14400)
def fetch_twse_market_pe_pbr():
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
                if len(code) == 4 and code.isdigit():
                    try:
                        pe = float(pe_raw) if pe_raw and pe_raw != "-" else None
                        pb = float(pb_raw) if pb_raw and pb_raw != "-" else None
                        stocks.append({"code": code, "name": name, "pe_market": pe, "pb_market": pb, "market": "TW"})
                    except ValueError:
                        continue
            return pd.DataFrame(stocks)
    except Exception:
        pass
    return pd.DataFrame()

@st.cache_data(ttl=14400)
def fetch_tpex_market_pe_pbr():
    url = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_peratio_analysis"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            data = res.json()
            stocks = []
            for item in data:
                code = str(item.get("SecuritiesCompanyCode") or item.get("CompanyCode") or item.get("Code") or "").strip()
                name = str(item.get("CompanyName") or item.get("Name") or "").strip()
                pe_raw = str(item.get("PriceEarningRatio") or item.get("PEratio") or "").replace(",", "")
                pb_raw = str(item.get("PriceBookRatio") or item.get("PBratio") or "").replace(",", "")
                if len(code) == 4 and code.isdigit():
                    try:
                        pe = float(pe_raw) if pe_raw and pe_raw != "-" else None
                        pb = float(pb_raw) if pb_raw and pb_raw != "-" else None
                        stocks.append({"code": code, "name": name, "pe_market": pe, "pb_market": pb, "market": "TWO"})
                    except ValueError:
                        continue
            return pd.DataFrame(stocks)
    except Exception:
        pass
    return pd.DataFrame()

# ================= 單檔深入財報精算 =================
@st.cache_data(ttl=3600)
def deep_analyze_stock(code, name, market_type):
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

        # 0. 產業分類
        industry = get_stock_industry(code)

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
        pbr = round(curr_price / (equity / shares), 2) if (equity and shares and shares > 0) else 1.0

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
            "產業": industry,
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

# ================= 依產業分類排版分批發送至 Discord =================
def send_to_discord(webhook_url: str, selected_stocks: list):
    if not webhook_url or not selected_stocks:
        return False

    # 按產業分類歸納
    industry_groups = defaultdict(list)
    for s in selected_stocks:
        industry_groups[s["產業"]].append(s)

    # 產業圖示映射
    icon_map = {
        "半導體業": "⚡", "電子零組件業": "🔌", "電腦及週邊設備業": "💻",
        "通信網路業": "📡", "光電業": "💡", "電子通路業": "📦", "資訊服務業": "🖥️",
        "建材營造": "🏗️", "生技醫療業": "💊", "電機機械": "⚙️", "鋼鐵工業": "🔩",
        "化學工業": "🧪", "航運業": "🚢", "觀光餐旅": "✈️", "金融保險業": "🏦"
    }

    messages = []
    current_msg = f"📢 **【全台股收益價值雷達 - 產業鏈精選名單】（共 {len(selected_stocks)} 檔）**\n"

    for ind, items in sorted(industry_groups.items(), key=lambda x: len(x[1]), reverse=True):
        icon = icon_map.get(ind, "📁")
        sec_header = f"\n{icon} **【{ind}】（{len(items)} 檔）**\n──────────────────\n"
        
        # 若加上標題會爆字數，先另起新則
        if len(current_msg) + len(sec_header) > 1800:
            messages.append(current_msg)
            current_msg = sec_header
        else:
            current_msg += sec_header

        for item in items:
            stock_line = (
                f"• **{item['名稱']} ({item['代號']})** | 現價:`{item['現價']:.2f}`\n"
                f"  營益率:`{item['營益率(%)']:.1f}%` | PE:`{item['本益比(PE)']}x` | PBR:`{item['淨值比(PBR)']}x` | ROE:`{item['五年均ROE(%)']:.1f}%`\n"
            )
            if len(current_msg) + len(stock_line) > 1800:
                messages.append(current_msg)
                current_msg = f"{icon} **【{ind} 續】**\n──────────────────\n" + stock_line
            else:
                current_msg += stock_line

    if current_msg:
        messages.append(current_msg)

    all_success = True
    for msg in messages:
        try:
            res = requests.post(webhook_url.strip(), json={"content": msg}, timeout=8)
            if res.status_code != 204:
                all_success = False
            time.sleep(0.5)
        except Exception:
            all_success = False

    return all_success

# ================= UI 介面 =================
st.title("🏛️ 全台股收益價值投資自動篩選雷達")
st.caption("官方 OpenAPI 毫秒級全市場粗篩 ➔ 官方產業鏈自動歸類 ➔ yfinance 深度基本面精算 ➔ Discord 分類推播")

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
    st.header("🔔 Discord 聯動設置")
    webhook_url = st.text_input(
        "Discord Webhook URL",
        value=DEFAULT_WEBHOOK,
        type="password",
        help="由後台 Secrets 自動填入，亦可手動貼上"
    )
    
    if st.button("⚡ 測試 Discord 連線", use_container_width=True):
        if webhook_url:
            try:
                t_res = requests.post(webhook_url.strip(), json={"content": "🏓 測試成功！Streamlit 與 Discord 已完美連線！"}, timeout=5)
                if t_res.status_code == 204:
                    st.toast("✅ 測試訊息已成功送達 Discord！")
                else:
                    st.error(f"❌ 傳送失敗，狀態碼：{t_res.status_code}")
            except Exception as e:
                st.error(f"❌ 連線異常：{e}")
        else:
            st.warning("⚠️ 尚未讀取到 Webhook 網址。")

if st.button("🚀 開始全市場智慧掃描", type="primary", use_container_width=True):
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
        st.session_state["total_count"] = len(df_market)
        df_filtered = df_market[
            (df_market["pe_market"].notna()) &
            (df_market["pe_market"] > 0) &
            (df_market["pe_market"] <= max_pe) &
            (df_market["pb_market"].notna()) &
            (df_market["pb_market"] <= max_pbr)
        ].copy()
        
        candidates = df_filtered.to_dict(orient="records")
        st.session_state["candidate_count"] = len(candidates)
        
        progress = st.progress(0)
        status = st.empty()
        deep_results = []
        
        for i, row in enumerate(candidates):
            status.text(f"正在精算財報: {row['name']} ({row['code']}) [{i+1}/{len(candidates)}]...")
            res = deep_analyze_stock(row["code"], row["name"], row["market"])
            if res:
                if res["營益率(%)"] >= min_op_margin and res["五年均ROE(%)"] >= min_roe:
                    if not filter_peg or (filter_peg and res["_peg_num"] <= 1.2):
                        deep_results.append(res)
            progress.progress((i + 1) / len(candidates))
            
        progress.empty()
        status.empty()
        st.session_state["scan_results"] = deep_results

# ================= 結果展示與產業分類區 =================
if st.session_state["scan_results"] is not None:
    results = st.session_state["scan_results"]
    st.info(f"📊 全市場共掃描 **{st.session_state['total_count']}** 檔股票 ➔ 經初篩後鎖定 **{st.session_state['candidate_count']}** 檔進入精算！")
    
    if results:
        df_raw = pd.DataFrame(results)
        st.success(f"🎉 掃描完成！全市場最終挑選出 **{len(df_raw)}** 檔符合【收益價值投資】的高勝率卓越標的：")
        
        # 1. 產業統計概況卡片
        industry_counts = df_raw["產業"].value_counts()
        st.markdown("### 🏢 本次低估標的之產業分佈概況")
        
        # 視覺化產業分佈長條圖
        st.bar_chart(industry_counts)

        # 2. 產業互動過濾器
        all_industries = ["全部產業"] + list(industry_counts.index)
        selected_ind = st.multiselect("🎯 快速依特定產業過濾表格：", options=all_industries, default=["全部產業"])
        
        if "全部產業" in selected_ind or not selected_ind:
            display_df = df_raw.copy()
        else:
            display_df = df_raw[df_raw["產業"].isin(selected_ind)].copy()

        cols = ["代號", "名稱", "產業", "現價", "營益率(%)", "本益比(PE)", "淨值比(PBR)", "五年均ROE(%)", "自主PEG"]
        st.dataframe(display_df[cols], use_container_width=True)
        
        st.divider()
        # 3. 推播按鈕
        col_btn, _ = st.columns([2, 3])
        with col_btn:
            if st.button("📢 將此精選名單（按產業分類）推送至 Discord", type="secondary", use_container_width=True):
                target_url = webhook_url if webhook_url else DEFAULT_WEBHOOK
                if target_url:
                    # 推送目前過濾後的名單，若未過濾則推送全部
                    push_list = display_df.to_dict(orient="records")
                    with st.spinner("正在依產業排版並分批推送至 Discord..."):
                        if send_to_discord(target_url, push_list):
                            st.toast("✅ 已成功依產業分類推送至 Discord 頻道！")
                        else:
                            st.error("❌ 推送失敗，請確認 Webhook 網址是否正確。")
                else:
                    st.warning("⚠️ 請先在側邊欄填入 Webhook URL 才能發送。")
    else:
        st.warning("初篩出的候選股在深度財報檢驗中未達標，建議在左側放寬營益率或 ROE 門檻。")
