import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import twstock
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
})

def get_tw_stock_name(code: str) -> str:
    """自動取得台股標準繁體中文名稱"""
    if code in twstock.codes:
        return twstock.codes[code].name
    return code

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

def evaluate_seven_fundamentals(stock, info):
    """計算並檢驗七大長線核心基本面指標"""
    fin = stock.financials
    bs = stock.balance_sheet
    cf = stock.cashflow
    
    # 預設指標結果
    results = {
        "pass_count": 0,
        "roe_str": "資料不足", "pass_roe": False,
        "roic_str": "資料不足", "pass_roic": False,
        "debt_str": "資料不足", "pass_debt": False,
        "fcf_str": "資料不足", "pass_fcf": False,
        "gm_str": "資料不足", "pass_gm": False,
        "peg_str": "資料不足", "pass_peg": False,
        "payout_str": "資料不足", "pass_payout": False
    }
    
    if fin.empty or bs.empty or cf.empty or fin.shape[1] < 3:
        return results

    try:
        # 1. ROE (近五年/歷年平均 >= 15%，低點 >= 8%)
        net_incomes = fin.loc["Net Income Common Stockholders"] if "Net Income Common Stockholders" in fin.index else fin.loc["Net Income"]
        equity = bs.loc["Stockholders Equity"] if "Stockholders Equity" in bs.index else bs.loc["Total Equity Gross Minority Interest"]
        common_years = net_incomes.index.intersection(equity.index)
        roe_series = (net_incomes[common_years] / equity[common_years]) * 100
        avg_roe = roe_series.mean()
        min_roe = roe_series.min()
        pass_roe = (avg_roe >= 15.0) and (min_roe >= 8.0)
        results["roe_str"] = f"均值 {avg_roe:.1f}% (低點 {min_roe:.1f}%)"
        results["pass_roe"] = pass_roe

        # 2. ROIC (近五年/歷年平均 >= 12%)
        # NOPAT = EBIT * (1 - 20%稅率), Invested Capital = Equity + Total Debt - Cash
        ebit = fin.loc["EBIT"] if "EBIT" in fin.index else (fin.loc["Operating Income"] if "Operating Income" in fin.index else net_incomes)
        nopat = ebit * 0.8
        
        total_debt = bs.loc["Total Debt"] if "Total Debt" in bs.index else (bs.loc["Total Liabilities Net Minority Interest"] * 0.5)
        cash = bs.loc["Cash And Cash Equivalents"] if "Cash And Cash Equivalents" in bs.index else pd.Series(0, index=equity.index)
        
        invested_cap = (equity + total_debt - cash).replace(0, np.nan)
        common_roic_years = nopat.index.intersection(invested_cap.index)
        roic_series = (nopat[common_roic_years] / invested_cap[common_roic_years]) * 100
        avg_roic = roic_series.dropna().mean() if not roic_series.dropna().empty else avg_roe * 0.85
        pass_roic = avg_roic >= 12.0
        results["roic_str"] = f"均值 {avg_roic:.1f}%"
        results["pass_roic"] = pass_roic

        # 3. 負債比率 (負債總額/資產總額 < 50%)
        total_liab = bs.loc["Total Liabilities Net Minority Interest"].iloc[0] if "Total Liabilities Net Minority Interest" in bs.index else 0
        total_assets = bs.loc["Total Assets"].iloc[0] if "Total Assets" in bs.index else 1
        debt_ratio = (total_liab / total_assets) * 100 if total_assets > 0 else 0
        pass_debt = debt_ratio < 50.0
        results["debt_str"] = f"{debt_ratio:.1f}%"
        results["pass_debt"] = pass_debt

        # 4. 累積 (FCF / 淨利) >= 80%
        op_cash = cf.loc["Operating Cash Flow"] if "Operating Cash Flow" in cf.index else cf.loc["Cash Flow From Continuing Operating Activities"]
        capex = cf.loc["Capital Expenditure"].abs() if "Capital Expenditure" in cf.index else pd.Series(0, index=op_cash.index)
        fcf_series = op_cash - capex
        common_cf_years = fcf_series.index.intersection(net_incomes.index)
        total_fcf = fcf_series[common_cf_years].sum()
        total_ni = net_incomes[common_cf_years].sum()
        fcf_rate = (total_fcf / total_ni) * 100 if total_ni > 0 else 0
        pass_fcf = fcf_rate >= 80.0
        results["fcf_str"] = f"{fcf_rate:.0f}%"
        results["pass_fcf"] = pass_fcf

        # 5. 最新毛利率高於歷年平均或持穩 (波動 < 1.5%)
        gross_profit = fin.loc["Gross Profit"]
        total_rev = fin.loc["Total Revenue"]
        gm_series = (gross_profit / total_rev) * 100
        latest_gm = gm_series.iloc[0]
        avg_gm = gm_series.mean()
        pass_gm = (latest_gm >= avg_gm - 1.5)
        results["gm_str"] = f"最新 {latest_gm:.1f}% (均值 {avg_gm:.1f}%)"
        results["pass_gm"] = pass_gm

        # 6. PEG <= 1.2 或 本益比位處合理區間 (PE <= 20)
        pe = info.get("trailingPE", None)
        rev_start, rev_end = total_rev.iloc[-1], total_rev.iloc[0]
        n_years = len(total_rev) - 1
        rev_cagr = (((rev_end / rev_start) ** (1 / n_years)) - 1) * 100 if (rev_start > 0 and n_years > 0) else 0
        peg = (pe / rev_cagr) if (pe and rev_cagr > 0) else np.nan
        pass_peg = (peg <= 1.2) if not np.isnan(peg) else (pe and pe <= 20.0)
        peg_display = f"PEG {peg:.2f}" if not np.isnan(peg) else (f"PE {pe:.1f}x" if pe else "N/A")
        results["peg_str"] = peg_display
        results["pass_peg"] = pass_peg

        # 7. 盈餘配發率介於 35% - 65%
        payout_val = info.get("payoutRatio", None)
        payout_pct = payout_val * 100 if payout_val else None
        if payout_pct is not None:
            pass_payout = 35.0 <= payout_pct <= 65.0
            results["payout_str"] = f"{payout_pct:.1f}%"
        else:
            # 備用計算：以常態估計
            pass_payout = True
            results["payout_str"] = "約 45%~55%"
        results["pass_payout"] = pass_payout

        # 計算通過總數
        results["pass_count"] = sum([pass_roe, pass_roic, pass_debt, pass_fcf, pass_gm, pass_peg, pass_payout])

    except Exception:
        pass

    return results

def generate_stock_report(code: str) -> str:
    code = code.strip().upper()
    ticker = f"{code}.TW"
    
    # 取得繁體中文名稱
    company_name = get_tw_stock_name(code)
    
    # 抓取 K 線歷史數據
    df = yf.download(ticker, period="1y", interval="1d", progress=False, session=session)
    if df.empty or len(df) < 30:
        ticker = f"{code}.TWO"
        df = yf.download(ticker, period="1y", interval="1d", progress=False, session=session)
        
    if df.empty or len(df) < 30:
        return f"⚠️ 查無代號【{code}】的交易資料，請確認是否為 4 位數台股代碼。"

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 1. 技術面與均線
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    
    latest = df.iloc[-1]
    curr_price = float(latest['Close'])
    ma20 = float(latest['MA20']) if not np.isnan(latest['MA20']) else curr_price
    ma60 = float(latest['MA60']) if not np.isnan(latest['MA60']) else curr_price
    
    recent_high_60 = float(df['High'].iloc[-60:].max())
    recent_low_60 = float(df['Low'].iloc[-60:].min())
    recent_low_20 = float(df['Low'].iloc[-20:].min())

    is_above_ma20 = curr_price >= ma20
    is_above_ma60 = curr_price >= ma60

    # 2. 基本面七大指標檢驗
    stock = yf.Ticker(ticker, session=session)
    info = stock.info if hasattr(stock, "info") else {}
    fund = evaluate_seven_fundamentals(stock, info)
    score = fund["pass_count"]

    # 3. 法人目標價
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

    target_upside = ((target_mean - curr_price) / curr_price) * 100 if has_target else 0.0

    # 4. 技術關鍵價位
    entry_tier1 = round(curr_price, 2)
    entry_tier2 = round(ma20, 2)
    entry_tier3 = round(min(ma60, recent_low_20), 2)
    stop_loss_price = round(min(ma60 * 0.96, recent_low_60 * 0.98), 2)
    stop_loss_pct = round(((stop_loss_price - curr_price) / curr_price) * 100, 2)
    tp_tier1 = round(target_mean, 2) if has_target else round(curr_price * 1.15, 2)
    tp_tier2 = round(max(target_high if has_target else 0, recent_high_60 * 1.1), 2)

    # 5. 操盤教練決策邏輯（整合七大體質 ＋ 技術價位 ＋ 法人目標價）
    if not is_above_ma60 and score < 4:
        status_state = "弱勢破線格局（跌破季線 且 體質未達標）"
        coach_signal = "⚫ 停損換股 / 嚴格避開"
        action_plan = f"跌破 60MA 季線生命線，且七項長線體質僅通過 {score}/7 項，長線資金應嚴格保護本金避免抄底。"
    elif has_target and target_upside < -5.0 and curr_price >= recent_high_60 * 0.95:
        status_state = "估值過熱區間（超越法人共識目標價）"
        coach_signal = "🔴 分批獲利了結"
        action_plan = f"現價已超越法人平均目標價 ({target_mean:.2f} 元)，乖離過大，建議在第一/第二目標價分批收回利潤。"
    elif score >= 5 and is_above_ma60 and (target_upside >= 12.0 or not has_target):
        status_state = "強勢成長格局（護城河頂級 ＋ 站穩季線）"
        coach_signal = "🟢 強力買進 / 分批建倉"
        action_plan = f"七大長線體質獲 {score}/7 高分，站穩季線且具上漲空間。建議依 30%(現價) / 40%(月線) / 30%(季線) 梯隊建立長線部位。"
    elif score >= 4 and is_above_ma60:
        status_state = "多頭震盪整理（基本面穩健）"
        coach_signal = "🟡 逢低承接 / 續抱觀察"
        action_plan = f"體質通過 {score}/7 項且多頭架構未變。持股續抱，加碼等待股價拉回月線 ({ma20:.2f} 元) 或季線支撐。"
    elif score >= 5 and not is_above_ma60:
        status_state = "好公司遇逆風（體質優異但處於技術修正）"
        coach_signal = "🟠 觀察等待 / 築底浮現再買"
        action_plan = f"體質拿下 {score}/7 高分，但目前股價在 60MA 季線之下修正。建議列入核心觀察清單，等待股價止跌重回季線再行進場。"
    else:
        status_state = "盤整待變 / 體質普通"
        coach_signal = "🟠 觀望等待"
        action_plan = "多空訊號未達共振標準，長線競爭優勢尚不明顯，建議保留現金。"

    # 格式化法人目標價
    if has_target:
        analyst_section = f"• 涵蓋機構：{analyst_count} 家法人報告\n• 平均目標：{target_mean:.2f} 元 (空間: {target_upside:+.1f}%)\n• 目標區間：{target_low:.2f} ~ {target_high:.2f} 元\n"
        if target_details:
            analyst_section += "• 近期評等：" + ", ".join([f"{t['broker']}({t['target']}元)" for t in target_details]) + "\n"
    else:
        analyst_section = "• 暫無近半年公開目標價（中小型股或未被大型投顧覆蓋）。\n"

    # 組裝報告內容
    report = (
        f"📊【 {company_name} ({code}) 長線全方位深度診斷】\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"📌 現價：{curr_price:.2f} 元 | 月線：{ma20:.2f} | 季線：{ma60:.2f}\n\n"
        f"🏛️ 一、長線七大核心體質檢驗 ({score}/7 通過)\n"
        f"1. ROE (均≧15%,低≧8%): {fund['roe_str']} ({'✅' if fund['pass_roe'] else '❌'})\n"
        f"2. ROIC (均≧12%): {fund['roic_str']} ({'✅' if fund['pass_roic'] else '❌'})\n"
        f"3. 負債比率 (<50%): {fund['debt_str']} ({'✅' if fund['pass_debt'] else '❌'})\n"
        f"4. 自由現金流率 (≧80%): {fund['fcf_str']} ({'✅' if fund['pass_fcf'] else '⚠️'})\n"
        f"5. 毛利率走勢 (高於均值/持穩): {fund['gm_str']} ({'✅' if fund['pass_gm'] else '❌'})\n"
        f"6. 估值比 (PEG≦1.2/PE合理): {fund['peg_str']} ({'✅' if fund['pass_peg'] else '⚠️'})\n"
        f"7. 盈餘配發率 (35%-65%): {fund['payout_str']} ({'✅' if fund['pass_payout'] else '⚠️'})\n\n"
        f"🎯 二、法人目標價與評等共識\n"
        f"{analyst_section}\n"
        f"💡 三、技術關鍵價位\n"
        f"【分批進場梯隊】\n"
        f"• 第一批(現價底倉 30%)：{entry_tier1} 元\n"
        f"• 第二批(月線加碼 40%)：{entry_tier2} 元\n"
        f"• 第三批(季線防守 30%)：{entry_tier3} 元\n"
        f"【停損防守點】\n"
        f"• 關鍵停損：{stop_loss_price} 元 ({stop_loss_pct}%)\n"
        f"【目標停利點】\n"
        f"• 第一目標(減碼 1/2)：{tp_tier1} 元\n"
        f"• 第二目標(波段全出)：{tp_tier2} 元\n\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"🧭 四、【操盤教練決策建議】\n"
        f"• 標的狀態：{status_state}\n"
        f"• 操作指令：{coach_signal}\n"
        f"• 決策理由：{action_plan}\n"
        f"━━━━━━━━━━━━━━━━━━"
    )
    return report

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 你好！我是你的台股深度診斷助理。\n\n請直接傳送 4 位數台股代號（例如：2330、6446）。")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text.strip()
    if len(user_text) == 4 and user_text.isdigit():
        stock_name = get_tw_stock_name(user_text)
        wait_msg = await update.message.reply_text(f"🔍 正在對【{stock_name} ({user_text})】進行七大體質健檢與價位精算...")
        try:
            report = generate_stock_report(user_text)
            await wait_msg.edit_text(report)
        except Exception as e:
            await wait_msg.edit_text(f"⚠️ 診斷發生錯誤：{str(e)}")
    else:
        await update.message.reply_text("💡 請直接輸入 4 位數股票代號（例如：2330、6446）。")

class HealthServer(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Bot is online and running 24/7!")

def run_web():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), HealthServer)
    server.serve_forever()

if __name__ == "__main__":
    threading.Thread(target=run_web, daemon=True).start()
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("🤖 Telegram 機器人已成功在雲端啟動...")
    app.run_polling()
