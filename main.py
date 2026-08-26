import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

# 建立偽裝瀏覽器的 Session 降低被阻擋機率
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
})

def generate_stock_report(code: str) -> str:
    code = code.strip().upper()
    ticker = f"{code}.TW"
    
    # 下載近 1 年 K 線歷史資料
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

    # 2. 基本面與目標價（加入 Try-Catch 避免 429 崩潰）
    company_name = code
    target_mean = None
    target_high = None
    target_low = None
    analyst_count = 0
    score = 2
    avg_roe, fcf_rate, latest_gm = 15.0, 80.0, 30.0
    pass_roe, pass_fcf, gm_trend_up = True, True, True

    try:
        stock = yf.Ticker(ticker, session=session)
        info = stock.info
        company_name = info.get("longName", info.get("shortName", code))
        target_mean = info.get("targetMeanPrice", None)
        target_high = info.get("targetHighPrice", None)
        target_low = info.get("targetLowPrice", None)
        analyst_count = info.get("numberOfAnalystOpinions", 0)
    except Exception:
        pass

    has_target = target_mean is not None and target_mean > 0
    target_upside = ((target_mean - curr_price) / curr_price) * 100 if has_target else 0.0

    # 3. 關鍵價位
    entry_tier1 = round(curr_price, 2)
    entry_tier2 = round(ma20, 2)
    entry_tier3 = round(min(ma60, recent_low_20), 2)
    stop_loss_price = round(min(ma60 * 0.96, recent_low_60 * 0.98), 2)
    stop_loss_pct = round(((stop_loss_price - curr_price) / curr_price) * 100, 2)
    tp_tier1 = round(target_mean, 2) if has_target else round(curr_price * 1.15, 2)
    tp_tier2 = round(max(target_high, recent_high_60 * 1.1), 2) if has_target else round(recent_high_60 * 1.10, 2)

    # 4. 教練建議
    if not is_above_ma60:
        status_state = "弱勢破線格局"
        coach_signal = "⚫ 停損換股 / 嚴格避開"
        action_plan = "跌破 60MA 季線生命線，長線資金應嚴格保護本金避免抄底。"
    elif has_target and target_upside < -5.0:
        status_state = "估值過熱區間"
        coach_signal = "🔴 分批獲利了結"
        action_plan = f"現價已超越法人平均目標價 ({target_mean:.2f} 元)，建議分批收回利潤。"
    elif is_above_ma60 and (target_upside >= 15.0 or not has_target):
        status_state = "強勢成長格局"
        coach_signal = "🟢 強力買進 / 分批建倉"
        action_plan = "站穩 60MA 季線之上且具備上漲空間，建議按 30%/40%/30% 梯隊建倉。"
    else:
        status_state = "多頭震盪整理"
        coach_signal = "🟡 逢低承接 / 續抱觀察"
        action_plan = f"多頭架構未變。持股續抱，加碼等待回測月線 ({ma20:.2f} 元) 或季線支撐。"

    analyst_section = f"• 涵蓋機構：{analyst_count} 家\n• 平均目標：{target_mean:.2f} 元 (空間: {target_upside:+.1f}%)\n• 目標區間：{target_low:.2f} ~ {target_high:.2f} 元\n" if has_target else "• 暫無法人公開目標價（中小型股或未被大型券商覆蓋）。\n"

    report = (
        f"📊【{company_name} ({code}) 深度診斷報告】\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"📌 現價：{curr_price:.2f} 元 | 月線：{ma20:.2f} | 季線：{ma60:.2f}\n\n"
        f"一、🎯 法人目標價共識\n"
        f"{analyst_section}\n"
        f"二、💡 技術關鍵價位\n"
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
        f"三、🧭 【操盤教練決策建議】\n"
        f"• 標的狀態：{status_state}\n"
        f"• 操作指令：{coach_signal}\n"
        f"• 決策理由：{action_plan}\n"
        f"━━━━━━━━━━━━━━━━━━"
    )
    return report

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 你好！我是你的台股深度診斷助理。\n\n請直接傳送 4 位數台股代號（例如：2330、3653）。")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text.strip()
    if len(user_text) == 4 and user_text.isdigit():
        wait_msg = await update.message.reply_text(f"🔍 正在調取【{user_text}】最新行情與法人目標價...")
        try:
            report = generate_stock_report(user_text)
            await wait_msg.edit_text(report)
        except Exception as e:
            await wait_msg.edit_text(f"⚠️ 診斷發生錯誤：{str(e)}")
    else:
        await update.message.reply_text("💡 請直接輸入 4 位數股票代號（例如：2330、3653）。")

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
