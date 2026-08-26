import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import yfinance as yf
import pandas as pd
import numpy as np
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

def generate_stock_report(code: str) -> str:
    code = code.strip().upper()
    ticker = f"{code}.TW"
    stock = yf.Ticker(ticker)
    info = stock.info
    
    if not info or ('regularMarketPrice' not in info and 'currentPrice' not in info):
        ticker = f"{code}.TWO"
        stock = yf.Ticker(ticker)
        info = stock.info
        
    curr_price = float(info.get("currentPrice", info.get("regularMarketPrice", 0.0)))
    company_name = info.get("longName", info.get("shortName", code))
    
    if curr_price == 0.0:
        return f"⚠️ 查無代號【{code}】的行情報價，請確認是否為 4 位數台股代碼。"

    # 1. 技術面
    df = stock.history(period="1y", interval="1d")
    if df.empty or len(df) < 60:
        return f"⚠️ 【{company_name} ({code})】歷史資料不足。"

    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    
    latest = df.iloc[-1]
    ma20 = float(latest['MA20'])
    ma60 = float(latest['MA60'])
    recent_high_60 = float(df['High'].iloc[-60:].max())
    recent_low_60 = float(df['Low'].iloc[-60:].min())
    recent_low_20 = float(df['Low'].iloc[-20:].min())

    is_above_ma20 = curr_price >= ma20
    is_above_ma60 = curr_price >= ma60

    # 2. 基本面
    fin, bs, cf = stock.financials, stock.balance_sheet, stock.cashflow
    pass_roe, pass_fcf, gm_trend_up, pass_val = False, False, False, False
    avg_roe, fcf_rate, latest_gm, oldest_gm, rev_cagr, peg = 0, 0, 0, 0, 0, np.nan
    pe = info.get("trailingPE", None)

    if not (fin.empty or bs.empty or cf.empty) and fin.shape[1] >= 3:
        net_incomes = fin.loc["Net Income Common Stockholders"] if "Net Income Common Stockholders" in fin.index else fin.loc["Net Income"]
        equity = bs.loc["Stockholders Equity"] if "Stockholders Equity" in bs.index else bs.loc["Total Equity Gross Minority Interest"]
        common_years = net_incomes.index.intersection(equity.index)
        roe_series = (net_incomes[common_years] / equity[common_years]) * 100
        avg_roe = roe_series.mean()
        pass_roe = (avg_roe >= 15.0) and (roe_series.min() >= 10.0)

        op_cash = cf.loc["Operating Cash Flow"] if "Operating Cash Flow" in cf.index else cf.loc["Cash Flow From Continuing Operating Activities"]
        capex = cf.loc["Capital Expenditure"].abs() if "Capital Expenditure" in cf.index else pd.Series(0, index=op_cash.index)
        fcf_series = op_cash - capex
        common_cf_years = fcf_series.index.intersection(net_incomes.index)
        total_fcf = fcf_series[common_cf_years].sum()
        total_ni = net_incomes[common_cf_years].sum()
        fcf_rate = (total_fcf / total_ni) * 100 if total_ni > 0 else 0
        pass_fcf = fcf_rate >= 80.0

        gross_profit, total_rev = fin.loc["Gross Profit"], fin.loc["Total Revenue"]
        gm_series = (gross_profit / total_rev) * 100
        latest_gm, oldest_gm = gm_series.iloc[0], gm_series.iloc[-1]
        gm_trend_up = (latest_gm >= oldest_gm) or (latest_gm >= gm_series.iloc[1] - 1.5)

        rev_start, rev_end = total_rev.iloc[-1], total_rev.iloc[0]
        n_years = len(total_rev) - 1
        rev_cagr = (((rev_end / rev_start) ** (1 / n_years)) - 1) * 100 if (rev_start > 0 and n_years > 0) else 0
        peg = (pe / rev_cagr) if (pe and rev_cagr > 0) else np.nan
        pass_val = (peg <= 1.2) if not np.isnan(peg) else (pe and pe <= 20)

    score = sum([pass_roe, pass_fcf, gm_trend_up, pass_val])

    # 3. 法人目標價
    target_mean = info.get("targetMeanPrice", None)
    target_high = info.get("targetHighPrice", None)
    target_low = info.get("targetLowPrice", None)
    analyst_count = info.get("numberOfAnalystOpinions", 0)
    has_target = target_mean is not None and target_mean > 0
    target_upside = ((target_mean - curr_price) / curr_price) * 100 if has_target else 0.0

    # 4. 關鍵價位
    entry_tier1 = round(curr_price, 2)
    entry_tier2 = round(ma20, 2)
    entry_tier3 = round(min(ma60, recent_low_20), 2)
    stop_loss_price = round(min(ma60 * 0.96, recent_low_60 * 0.98), 2)
    stop_loss_pct = round(((stop_loss_price - curr_price) / curr_price) * 100, 2)
    tp_tier1 = round(target_mean, 2) if has_target else round(curr_price * 1.15, 2)
    tp_tier2 = round(max(target_high, recent_high_60 * 1.1), 2) if has_target else round(recent_high_60 * 1.10, 2)

    # 5. 教練建議
    if not is_above_ma60 and score < 2:
        status_state = "弱勢破線格局"
        coach_signal = "⚫ 停損換股 / 嚴格避開"
        action_plan = "跌破 60MA 季線且財報評分低，應嚴格保護本金避免抄底。"
    elif has_target and target_upside < -5.0 and curr_price >= recent_high_60 * 0.95:
        status_state = "估值過熱區間"
        coach_signal = "🔴 分批獲利了結"
        action_plan = f"現價已高於法人平均目標價 ({target_mean:.2f} 元)，建議分批收回利潤。"
    elif score >= 3 and is_above_ma60 and (target_upside >= 15.0 or not has_target):
        status_state = "強勢成長格局"
        coach_signal = "🟢 強力買進 / 分批建倉"
        action_plan = f"基本面獲 {score}/4 分且站穩季線，建議按 30%/40%/30% 梯隊建倉。"
    elif score >= 2 and is_above_ma60:
        status_state = "多頭震盪整理"
        coach_signal = "🟡 逢低承接 / 續抱觀察"
        action_plan = f"多頭架構未變。續抱或等待回測月線 ({ma20:.2f} 元) 支撐加碼。"
    else:
        status_state = "盤整待變"
        coach_signal = "🟠 觀望等待"
        action_plan = "多空訊號未達標準共振，列入自選等待更佳買點。"

    analyst_section = f"• 涵蓋機構：{analyst_count} 家\n• 平均目標：{target_mean:.2f} 元 (空間: {target_upside:+.1f}%)\n• 目標區間：{target_low:.2f} ~ {target_high:.2f} 元\n" if has_target else "• 暫無法人公開目標價覆蓋。\n"

    report = (
        f"📊【{company_name} ({code}) 深度診斷報告】\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"📌 現價：{curr_price:.2f} 元 | 月線：{ma20:.2f} | 季線：{ma60:.2f}\n\n"
        f"一、基本面護城河 ({score}/4 分)\n"
        f"• 歷年平均 ROE：{avg_roe:.1f}% ({'✅' if pass_roe else '❌'})\n"
        f"• 自由現金流率：{fcf_rate:.1f}% ({'✅' if pass_fcf else '⚠️'})\n"
        f"• 毛利率趨勢：最新 {latest_gm:.1f}% ({'🟢 向上' if gm_trend_up else '🔴 衰退'})\n\n"
        f"二、🎯 法人目標價共識\n"
        f"{analyst_section}\n"
        f"三、💡 技術關鍵價位\n"
        f"【分批進場】\n"
        f"• 第一批(現價底倉)：{entry_tier1} 元\n"
        f"• 第二批(月線加碼)：{entry_tier2} 元\n"
        f"• 第三批(季線防守)：{entry_tier3} 元\n"
        f"【停損防守點】\n"
        f"• 建議停損：{stop_loss_price} 元 ({stop_loss_pct}%)\n"
        f"【目標停利點】\n"
        f"• 第一目標(減碼 1/2)：{tp_tier1} 元\n"
        f"• 第二目標(波段全出)：{tp_tier2} 元\n\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"四、🧭 【操盤教練決策建議】\n"
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
        wait_msg = await update.message.reply_text(f"🔍 正在調取【{user_text}】最新財報與法人目標價...")
        try:
            report = generate_stock_report(user_text)
            await wait_msg.edit_text(report)
        except Exception as e:
            await wait_msg.edit_text(f"⚠️ 診斷發生錯誤：{str(e)}")
    else:
        await update.message.reply_text("💡 請直接輸入 4 位數股票代號（例如：2330、2454）。")

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
