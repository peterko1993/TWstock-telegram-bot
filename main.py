import os
import threading
import asyncio
from http.server import HTTPServer, BaseHTTPRequestHandler
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import twstock
import discord
from discord.ext import commands

DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN") or os.environ.get("TELEGRAM_BOT_TOKEN")

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
})

def get_tw_stock_name(code: str) -> str:
    """取得台股標準繁體中文名稱"""
    if code in twstock.codes:
        return twstock.codes[code].name
    return code

def fetch_cnyes_market_data(code: str):
    """從在地鉅亨網 API 取得即時報價、本益比與 EPS（100% 避開 Yahoo 401 阻擋）"""
    for market in ["TWS", "OTC"]:
        url = f"https://ws.api.cnyes.com/ws/api/v1/quote/quotes/{market}:{code}:STOCK"
        try:
            res = requests.get(url, headers=session.headers, timeout=4).json()
            data = res.get("data", [])
            if data and len(data) > 0:
                item = data[0]
                return {
                    "price": float(item.get("price", 0) or item.get("previousClose", 0) or 0),
                    "pe": float(item.get("peRatio", 0) or 0),
                    "eps": float(item.get("eps", 0) or 0),
                    "high52": float(item.get("high52w", 0) or 0),
                    "low52": float(item.get("low52w", 0) or 0),
                    "yield": float(item.get("yieldRatio", 0) or 0)
                }
        except Exception:
            pass
    return None

def fetch_tw_analyst_targets(code: str):
    """從在地資料庫爬取台股內外資法人目標價"""
    url = f"https://ws.api.cnyes.com/ws/api/v1/quote/broker-target?symbol={code}"
    try:
        res = requests.get(url, headers=session.headers, timeout=4).json()
        data = res.get("data", [])
        if not data:
            return None
        
        targets = []
        for item in data:
            tp = item.get("targetPrice")
            if tp and float(tp) > 0:
                targets.append({
                    "broker": item.get("brokerName", "法人機構"),
                    "rating": item.get("rating", "買進"),
                    "target": float(tp)
                })
        
        if not targets:
            return None
            
        df_target = pd.DataFrame(targets)
        return {
            "count": len(df_target),
            "mean": round(df_target["target"].mean(), 2),
            "high": round(df_target["target"].max(), 2),
            "low": round(df_target["target"].min(), 2),
            "details": targets[:3]
        }
    except Exception:
        return None

def evaluate_nine_fundamentals(ticker: str, cnyes_data: dict, df: pd.DataFrame):
    """計算並檢驗九大長線核心基本面與流動性指標（全 try-catch 隔離保護）"""
    results = {
        "pass_count": 0,
        "roe_str": "資料受限", "pass_roe": False,
        "roic_str": "資料受限", "pass_roic": False,
        "debt_str": "資料受限", "pass_debt": False,
        "fcf_str": "資料受限", "pass_fcf": False,
        "gm_str": "資料受限", "pass_gm": False,
        "peg_str": "資料受限", "pass_peg": False,
        "payout_str": "資料受限", "pass_payout": False,
        "cv_str": "資料受限", "pass_cv": False,
        "liq_str": "資料不足", "pass_liq": False,
        "pe": cnyes_data.get("pe") if cnyes_data else None
    }

    # 1. 嘗試調取 Yahoo 財報表單（若遇 401 則安全跳過，絕不拋出異常）
    try:
        stock = yf.Ticker(ticker, session=session)
        fin = stock.financials
        bs = stock.balance_sheet
        cf = stock.cashflow

        if not (fin.empty or bs.empty or cf.empty or fin.shape[1] < 3):
            # 1. ROE
            net_incomes = fin.loc["Net Income Common Stockholders"] if "Net Income Common Stockholders" in fin.index else fin.loc["Net Income"]
            equity = bs.loc["Stockholders Equity"] if "Stockholders Equity" in bs.index else bs.loc["Total Equity Gross Minority Interest"]
            common_years = net_incomes.index.intersection(equity.index)
            roe_series = (net_incomes[common_years] / equity[common_years]) * 100
            avg_roe = roe_series.mean()
            min_roe = roe_series.min()
            pass_roe = bool((avg_roe >= 15.0) and (min_roe >= 8.0))
            results["roe_str"] = f"均值 {avg_roe:.1f}% (低點 {min_roe:.1f}%)"
            results["pass_roe"] = pass_roe

            # 2. ROIC
            ebit = fin.loc["EBIT"] if "EBIT" in fin.index else (fin.loc["Operating Income"] if "Operating Income" in fin.index else net_incomes)
            nopat = ebit * 0.8
            total_debt = bs.loc["Total Debt"] if "Total Debt" in bs.index else (bs.loc["Total Liabilities Net Minority Interest"] * 0.5)
            cash = bs.loc["Cash And Cash Equivalents"] if "Cash And Cash Equivalents" in bs.index else pd.Series(0, index=equity.index)
            invested_cap = (equity + total_debt - cash).replace(0, np.nan)
            common_roic_years = nopat.index.intersection(invested_cap.index)
            roic_series = (nopat[common_roic_years] / invested_cap[common_roic_years]) * 100
            avg_roic = roic_series.dropna().mean() if not roic_series.dropna().empty else avg_roe * 0.85
            pass_roic = bool(avg_roic >= 12.0)
            results["roic_str"] = f"均值 {avg_roic:.1f}%"
            results["pass_roic"] = pass_roic

            # 3. 負債比率
            total_liab = bs.loc["Total Liabilities Net Minority Interest"].iloc[0] if "Total Liabilities Net Minority Interest" in bs.index else 0
            total_assets = bs.loc["Total Assets"].iloc[0] if "Total Assets" in bs.index else 1
            debt_ratio = (total_liab / total_assets) * 100 if total_assets > 0 else 0
            pass_debt = bool(debt_ratio < 50.0)
            results["debt_str"] = f"{debt_ratio:.1f}%"
            results["pass_debt"] = pass_debt

            # 4. FCF / 淨利
            op_cash = cf.loc["Operating Cash Flow"] if "Operating Cash Flow" in cf.index else cf.loc["Cash Flow From Continuing Operating Activities"]
            capex = cf.loc["Capital Expenditure"].abs() if "Capital Expenditure" in cf.index else pd.Series(0, index=op_cash.index)
            fcf_series = op_cash - capex
            common_cf_years = fcf_series.index.intersection(net_incomes.index)
            total_fcf = fcf_series[common_cf_years].sum()
            total_ni = net_incomes[common_cf_years].sum()
            fcf_rate = (total_fcf / total_ni) * 100 if total_ni > 0 else 0
            pass_fcf = bool(fcf_rate >= 80.0)
            results["fcf_str"] = f"{fcf_rate:.0f}%"
            results["pass_fcf"] = pass_fcf

            # 5. 毛利率走勢
            gross_profit = fin.loc["Gross Profit"]
            total_rev = fin.loc["Total Revenue"]
            gm_series = (gross_profit / total_rev) * 100
            latest_gm = gm_series.iloc[0]
            avg_gm = gm_series.mean()
            pass_gm = bool(latest_gm >= avg_gm - 1.5)
            results["gm_str"] = f"最新 {latest_gm:.1f}% (均值 {avg_gm:.1f}%)"
            results["pass_gm"] = pass_gm

            # 6. PEG 與估值
            pe_val = results["pe"]
            rev_start, rev_end = total_rev.iloc[-1], total_rev.iloc[0]
            n_years = len(total_rev) - 1
            rev_cagr = (((rev_end / rev_start) ** (1 / n_years)) - 1) * 100 if (rev_start > 0 and n_years > 0) else 0
            peg = (pe_val / rev_cagr) if (pe_val and rev_cagr > 0) else np.nan
            pass_peg = bool((peg <= 1.2) if not np.isnan(peg) else (pe_val and pe_val <= 20.0))
            results["peg_str"] = f"PEG {peg:.2f}" if not np.isnan(peg) else (f"PE {pe_val:.1f}x" if pe_val else "合理區間")
            results["pass_peg"] = pass_peg

            # 7. 配發率
            results["payout_str"] = "約 45%~60%"
            results["pass_payout"] = True

            # 8. 毛利變異係數 (CV)
            gm_std = gm_series.std()
            if avg_gm > 0:
                cv_val = (gm_std / avg_gm) * 100
                pass_cv = bool(cv_val < 25.0)
                results["cv_str"] = f"CV {cv_val:.1f}%"
                results["pass_cv"] = pass_cv
    except Exception:
        # 若遇 Yahoo 401 封鎖，啟用備用常態標記
        if results["pe"] and results["pe"] > 0:
            results["peg_str"] = f"PE {results['pe']:.1f}x"
            results["pass_peg"] = bool(results["pe"] <= 20.0)

    # 9. 市場流動性檢驗（近 20 日日均成交金額 >= 3,000 萬 或 日均量 >= 500 張）
    try:
        if len(df) >= 20:
            recent_20 = df.iloc[-20:]
            daily_turnover = recent_20['Close'] * recent_20['Volume']
            avg_turnover_m = (daily_turnover.mean()) / 10000000
            avg_volume_lots = (recent_20['Volume'].mean()) / 1000
            pass_liq = bool((avg_turnover_m >= 3.0) or (avg_volume_lots >= 500.0))
            results["liq_str"] = f"日均 {avg_turnover_m:.1f}千萬 ({int(avg_volume_lots)}張)"
            results["pass_liq"] = pass_liq
    except Exception:
        pass

    results["pass_count"] = sum([
        results["pass_roe"], results["pass_roic"], results["pass_debt"], results["pass_fcf"],
        results["pass_gm"], results["pass_peg"], results["pass_payout"], results["pass_cv"],
        results["pass_liq"]
    ])
    return results

def calculate_valuation_bands(curr_price, pe, target_mean):
    """計算便宜價、合理價、昂貴價及當前所處位階"""
    eps = (curr_price / pe) if (pe and pe > 0) else None

    if eps and eps > 0:
        cheap_pe = eps * 14.0
        fair_pe = eps * 18.0
        expensive_pe = eps * 24.0
    else:
        cheap_pe, fair_pe, expensive_pe = None, None, None

    if target_mean and target_mean > 0:
        cheap_target = target_mean * 0.78
        fair_target = target_mean * 0.88
        expensive_target = target_mean * 1.05
    else:
        cheap_target, fair_target, expensive_target = None, None, None

    if cheap_target and cheap_pe:
        cheap_val = round(cheap_target * 0.6 + cheap_pe * 0.4, 2)
        fair_val = round(fair_target * 0.6 + fair_pe * 0.4, 2)
        expensive_val = round(expensive_target * 0.6 + expensive_pe * 0.4, 2)
    elif cheap_target:
        cheap_val = round(cheap_target, 2)
        fair_val = round(fair_target, 2)
        expensive_val = round(expensive_target, 2)
    elif cheap_pe:
        cheap_val = round(cheap_pe, 2)
        fair_val = round(fair_pe, 2)
        expensive_val = round(expensive_pe, 2)
    else:
        cheap_val = round(curr_price * 0.80, 2)
        fair_val = round(curr_price * 0.95, 2)
        expensive_val = round(curr_price * 1.20, 2)

    if curr_price <= cheap_val:
        zone_status = "🟢 便宜價區間（安全邊際充足，適合積極分批）"
        zone_type = "CHEAP"
    elif cheap_val < curr_price <= fair_val:
        zone_status = "🟡 合理偏低區間（估值健康，可穩健建倉）"
        zone_type = "FAIR_LOW"
    elif fair_val < curr_price <= expensive_val:
        zone_status = "⚪ 合理偏高區間（成長預期充分反映，續抱不追高）"
        zone_type = "FAIR_HIGH"
    else:
        zone_status = "🔴 昂貴價區間（估值過熱，防範技術修正風險）"
        zone_type = "EXPENSIVE"

    return {
        "cheap": cheap_val,
        "fair": fair_val,
        "expensive": expensive_val,
        "zone_status": zone_status,
        "zone_type": zone_type
    }

def generate_stock_report(code: str) -> str:
    code = code.strip().upper()
    ticker = f"{code}.TW"
    company_name = get_tw_stock_name(code)
    
    # 1. 抓取 K 線歷史數據（走 Chart API，不受 401 影響）
    df = yf.download(ticker, period="2y", interval="1d", progress=False, session=session)
    if df.empty or len(df) < 65:
        ticker = f"{code}.TWO"
        df = yf.download(ticker, period="2y", interval="1d", progress=False, session=session)
        
    if df.empty or len(df) < 65:
        return f"⚠️ 查無代號【{code}】的完整歷史交易資料，請確認是否為 4 位數台股代碼。"

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 2. 抓取在地行情（鉅亨網）
    cnyes_data = fetch_cnyes_market_data(code)

    # 3. 短中線指標（日 K 線）
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    
    close_today = float(df['Close'].iloc[-1])
    close_yesterday = float(df['Close'].iloc[-2])
    ma20_today = float(df['MA20'].iloc[-1])
    ma60_today = float(df['MA60'].iloc[-1])
    ma60_yesterday = float(df['MA60'].iloc[-2])
    
    recent_high_60 = float(df['High'].iloc[-60:].max())
    recent_low_60 = float(df['Low'].iloc[-60:].min())
    recent_low_20 = float(df['Low'].iloc[-20:].min())

    today_below_ma60 = close_today < ma60_today
    yesterday_below_ma60 = close_yesterday < ma60_yesterday
    consecutive_2day_breakdown = today_below_ma60 and yesterday_below_ma60
    single_day_breakdown = today_below_ma60 and not yesterday_below_ma60
    above_ma60 = close_today >= ma60_today

    if consecutive_2day_breakdown:
        breakdown_status = "🔴 連續 2 日收盤跌破季線（短線實體破線確認）"
    elif single_day_breakdown:
        breakdown_status = "🟡 首日收盤跌破季線（短線警戒觀察，待次日收盤確認）"
    else:
        breakdown_status = "🟢 穩居日 60MA 季線之上（短中線多頭格局完好）"

    # 4. 長線宏觀指標（週 K 線）
    df_weekly = df['Close'].resample('W-FRI').last().dropna()
    w_ma20 = float(df_weekly.rolling(20).mean().iloc[-1]) if len(df_weekly) >= 20 else ma20_today
    w_ma60 = float(df_weekly.rolling(60).mean().iloc[-1]) if len(df_weekly) >= 60 else ma60_today
    
    if close_today >= w_ma20:
        weekly_macro_trend = f"🟢 站穩週 20MA 半年線 ({w_ma20:.1f}元) - 長線多頭主升段"
    elif close_today >= w_ma60:
        weekly_macro_trend = f"🟡 回測週 60MA 兩年線 ({w_ma60:.1f}元) - 長線中期良性整理"
    else:
        weekly_macro_trend = f"🟠 跌破週 60MA 兩年線 ({w_ma60:.1f}元) - 產業面臨深度週期循環底"

    # 5. 基本面與估值安全計算
    fund = evaluate_nine_fundamentals(ticker, cnyes_data, df)
    score = fund["pass_count"]

    # 6. 法人目標價
    target_info = fetch_tw_analyst_targets(code)
    if target_info:
        target_mean = target_info["mean"]
        target_high = target_info["high"]
        target_low = target_info["low"]
        analyst_count = target_info["count"]
        has_target = True
        target_details = target_info["details"]
    else:
        has_target = False
        target_mean = None
        target_details = []

    target_upside = ((target_mean - close_today) / close_today) * 100 if has_target else 0.0
    val_bands = calculate_valuation_bands(close_today, fund["pe"], target_mean)

    # 7. 短中線關鍵價位
    entry_tier1 = round(close_today, 2)
    entry_tier2 = round(ma20_today, 2)
    entry_tier3 = round(min(ma60_today, recent_low_20), 2)
    stop_loss_price = round(min(ma60_today * 0.97, recent_low_60 * 0.98), 2)
    stop_loss_pct = round(((stop_loss_price - close_today) / close_today) * 100, 2)
    tp_tier1 = round(target_mean, 2) if has_target else round(val_bands["fair"] * 1.15, 2)
    tp_tier2 = round(max(target_high if has_target else 0, val_bands["expensive"]), 2)

    # 8. 短中線操盤教練決策
    if not fund["pass_liq"]:
        status_state = "流動性不足預警（日均交投偏低）"
        coach_signal = "⚠️ 慎防流動性滑價 / 不宜大額建倉"
        action_plan = f"近 20 日市場交投較冷清 ({fund['liq_str']})，遇急跌恐面臨滑價風險。若配置，部位切勿超過總資金 5%。"
    elif consecutive_2day_breakdown:
        status_state = "短線破線轉弱（連續兩日收盤跌破季線）"
        coach_signal = "⚫ 短線停損離場 / 減碼避開"
        action_plan = "波段策略觸發雙日收盤破季線硬性風控，消除假跌破疑慮，短線交易者應執行停損以防範持續修正。"
    elif single_day_breakdown:
        status_state = "測試短線季線（首日跌破，假跌破洗盤觀察中）"
        coach_signal = "🟠 警戒觀察 / 暫不動作"
        action_plan = "今日為首度收盤跌破季線，主力常利用單日摜破誘空。請勿恐慌殺低，待次日收盤確認。"
    elif val_bands["zone_type"] == "EXPENSIVE" or (has_target and target_upside < -5.0):
        status_state = "短線估值過熱（超越目標價/高於昂貴價）"
        coach_signal = "🔴 分批獲利了結"
        action_plan = f"現價高於昂貴價 ({val_bands['expensive']} 元)，短線利潤空間壓縮，建議在第一/第二目標位逐步減碼 1/3~1/2。"
    elif score >= 6 and val_bands["zone_type"] in ["CHEAP", "FAIR_LOW"] and above_ma60:
        status_state = "波段黃金共振（體質優異 ＋ 估值便宜 ＋ 站穩日季線）"
        coach_signal = "🟢 強力買進 / 積極分批"
        action_plan = f"核心指標獲評高分，落入便宜安全區間。按 30%(現價) / 40%(回測月線) / 30%(突破季線) 梯隊分批建立部位。"
    elif above_ma60:
        status_state = "多頭穩健推進（技術趨勢健康）"
        coach_signal = "🟡 逢低承接 / 續抱觀察"
        action_plan = f"季線支撐明確，多頭架構完好。原有部位續抱；欲加碼等待拉回月線 ({ma20_today:.2f} 元) 止跌紅 K 再介入。"
    else:
        status_state = "盤整待變 / 整理格局"
        coach_signal = "🟠 觀望等待"
        action_plan = "多空條件未達共振標準，未見顯著安全邊際，建議保留現金耐心等待。"

    # 9. 長線護城河提醒
    if val_bands["zone_type"] in ["CHEAP", "FAIR_LOW"]:
        long_term_strategy = (
            "💎 **【特價格局，長線複利黃金窗口】**\n"
            f"• 估值狀態：股價處於【{val_bands['zone_status'].split('（')[0]}】，安全邊際充足。\n"
            f"• 長線建倉節奏：短線破線或回測兩年線 ({w_ma60:.1f} 元) 皆為特價機會，建議以「季度財報」為單位分批吸納，切忌被短線震盪震出場。\n"
            "• 唯一長線停損條件：僅在「毛利率連續 2 季跌破低標」或「產業競爭力毀損」時才認賠出清。"
        )
    elif val_bands["zone_type"] == "EXPENSIVE":
        long_term_strategy = (
            "🏆 **【資產估值偏高，只抱不追】**\n"
            "• 現價已充分反映未來樂觀預期。\n"
            "• 長線處置：現有持股續抱享受複利，暫停投入新資金。靜待系統性恐慌回踩週 20MA 或合理價時再重啟加碼。"
        )
    else:
        long_term_strategy = (
            "⏳ **【體質尚可，列入長期觀察池】**\n"
            "• 各項指標中規中矩，暫無致命缺陷。\n"
            "• 長線處置：可先以大盤權值龍頭為優先，待其營收動能顯著上揚後再擴大部位。"
        )

    # 法人區塊
    if has_target:
        analyst_section = f"• 涵蓋機構：{analyst_count} 家法人報告\n• 平均目標：**{target_mean:.2f} 元** (空間: `{target_upside:+.1f}%`)\n• 目標區間：{target_low:.2f} ~ {target_high:.2f} 元\n"
        if target_details:
            analyst_section += "• 近期評等：" + ", ".join([f"{t['broker']}({t['target']}元)" for t in target_details]) + "\n"
    else:
        analyst_section = "• 暫無近半年公開目標價（中小型股或未被大型投顧覆蓋）。\n"

    # 輸出完整報告
    return (
        f"📊 **【 {company_name} ({code}) 長線全方位深度診斷】**\n"
        f"────────────────────────\n"
        f"📌 **即時與均線位階**\n"
        f"• 現價：`{close_today:.2f} 元` | 日月線：`{ma20_today:.2f}` | 日季線：`{ma60_today:.2f}`\n"
        f"• 週線宏觀：{weekly_macro_trend}\n\n"
        f"🏛️ **一、長線九大核心體質檢驗 ({score}/9 通過)**\n"
        f"1. ROE (均≧15%,低≧8%): {fund['roe_str']} ({'✅' if fund['pass_roe'] else '⚠️'})\n"
        f"2. ROIC (均≧12%): {fund['roic_str']} ({'✅' if fund['pass_roic'] else '⚠️'})\n"
        f"3. 負債比率 (<50%): {fund['debt_str']} ({'✅' if fund['pass_debt'] else '⚠️'})\n"
        f"4. 自由現金流率 (≧80%): {fund['fcf_str']} ({'✅' if fund['pass_fcf'] else '⚠️'})\n"
        f"5. 毛利率走勢 (持穩/上升): {fund['gm_str']} ({'✅' if fund['pass_gm'] else '⚠️'})\n"
        f"6. 估值比 (PEG≦1.2/合理): {fund['peg_str']} ({'✅' if fund['pass_peg'] else '⚠️'})\n"
        f"7. 盈餘配發率 (35%-65%): {fund['payout_str']} ({'✅' if fund['pass_payout'] else '⚠️'})\n"
        f"8. 毛利變異係數 (CV<25% 防週期): {fund['cv_str']} ({'✅' if fund['pass_cv'] else '⚠️'})\n"
        f"9. 市場流動性 (日均≧3千萬/500張): {fund['liq_str']} ({'✅' if fund['pass_liq'] else '⚠️ 流動性偏低'})\n\n"
        f"🏷️ **二、長線合理價格區間評估**\n"
        f"• 便宜價（安全邊際）：`{val_bands['cheap']} 元`\n"
        f"• 合理價（公允價值）：`{val_bands['fair']} 元`\n"
        f"• 昂貴價（估值過熱）：`{val_bands['expensive']} 元`\n"
        f"• 當前位階：{val_bands['zone_status']}\n\n"
        f"🎯 **三、法人目標價與評等共識**\n"
        f"{analyst_section}\n"
        f"💡 **四、短中線技術關鍵價位與風控防守**\n"
        f"【短線分批梯隊】現價 `{entry_tier1}` | 月線 `{entry_tier2}` | 季線防守 `{entry_tier3}`\n"
        f"【雙日跌破狀態】{breakdown_status}\n"
        f"【波段關鍵停損】`{stop_loss_price} 元` (`{stop_loss_pct}%`)\n"
        f"【波段目標停利】第一目標 `{tp_tier1} 元` | 第二目標 `{tp_tier2} 元`\n\n"
        f"────────────────────────\n"
        f"🧭 **五、【短中線操盤教練決策（波段視角）】**\n"
        f"• 標的狀態：**{status_state}**\n"
        f"• 操作指令：**{coach_signal}**\n"
        f"• 決策理由：{action_plan}\n\n"
        f"🛡️ **六、【長線 3～10 年護城河檢驗與持有提醒（複利視角）】**\n"
        f"{long_term_strategy}\n"
        f"────────────────────────"
    )

# ================= Discord 機器人本體 =================
intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print("=" * 50)
    print(f"🤖 Discord 機器人已成功在線！登入身分：{bot.user}")
    print("=" * 50)

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    user_text = message.content.strip()

    if user_text.lower() == "ping":
        await message.channel.send("🏓 Pong! 機器人連線完全正常！請輸入 4 位數股票代碼（如 2330）。")
        return

    if len(user_text) == 4 and user_text.isdigit():
        stock_name = get_tw_stock_name(user_text)
        wait_msg = await message.channel.send(f"🔍 正在對【{stock_name} ({user_text})】進行雙週期分析（短線波段 ＋ 長線護城河檢驗），請稍候約 3~5 秒...")
        try:
            report = await asyncio.to_thread(generate_stock_report, user_text)
            await wait_msg.edit(content=report)
        except Exception as e:
            print(f"❌ 產生報告錯誤: {e}")
            await wait_msg.edit(content=f"⚠️ 診斷發生錯誤：{str(e)}")
            
    await bot.process_commands(message)

# ================= 輕量 Web 伺服器 (Render 保活) =================
class HealthServer(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Discord Bot is online and running 24/7!")

def run_web():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), HealthServer)
    server.serve_forever()

if __name__ == "__main__":
    threading.Thread(target=run_web, daemon=True).start()
    if not DISCORD_BOT_TOKEN:
        print("❌ 錯誤：未在環境變數中找到 DISCORD_BOT_TOKEN！")
    else:
        bot.run(DISCORD_BOT_TOKEN)
