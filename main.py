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

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
})

def fetch_tw_analyst_targets(code: str):
    """從鉅亨網/在地資料庫爬取台股內外資法人目標價"""
    url = f"https://ws.api.cnyes.com/ws/api/v1/quote/broker-target?symbol={code}"
    try:
        res = requests.get(url, headers=session.headers, timeout=4).json()
        data = res.get("data", [])
        if not data:
            return None
        
        targets = []
        for item in data:
            # 取得近半年內的報告
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
        avg_target = df_target["target"].mean()
        high_target = df_target["target"].max()
        low_target = df_target["target"].min()
        
        return {
            "count": len(df_target),
            "mean": round(avg_target, 2),
            "high": round(high_target, 2),
            "low": round(low_target, 2),
            "details": targets[:3]  # 列出前 3 家近期法人
        }
    except Exception:
        return None

def generate_stock_report(code: str) -> str:
    code = code.strip().upper()
    ticker = f"{code}.TW"
    
    # 抓取 K 線歷史數據
    df = yf.download(ticker, period="1y", interval="1d", progress=False, session=session)
    if df.empty or len(df) < 30:
        ticker = f"{code}.TWO"
        df = yf.download(ticker, period="1y", interval="1d", progress=False, session=session)
        
    if df.empty or len(df) < 30:
        return f"⚠️ 查無代號【{code}】的交易資料，請確認是否為 4 位數台股代碼。"

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 1. 技術面
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

    # 2. 基本資訊
    company_name = code
    try:
        stock = yf.Ticker(ticker, session=session)
        company_name = stock.info.get("longName", stock.info.get("shortName", code))
    except Exception:
        pass

    # 3. 抓取法人目標價（優先使用台灣在地資料，備用 yfinance）
    target_info = fetch_tw_analyst_targets(code)
    
    if target_info:
        target_mean = target_info["mean"]
        target_high = target_info["high"]
        target_low = target_info["low"]
        analyst_count = target_info["count"]
        has_target = True
        target_details = target_info["details"]
    else:
        # 備用 yfinance 全球庫
        try:
            stock = yf.Ticker(ticker, session=session)
            info = stock.info
            target_mean = info.get("targetMeanPrice", None)
            target_high = info.get("targetHighPrice", None)
            target_low = info.get("targetLowPrice", None)
            analyst_count = info.get("numberOfAnalystOpinions", 0)
            has_target = target_mean is not None and target_mean > 0
            target_details = []
        except Exception:
            has_target = False
            target_mean = None
            target_details = []

    target_upside = ((target_mean - curr_price) / curr_price) * 100 if has_target else 0.0

    # 4. 關鍵價位
    entry_tier1 = round(curr_price, 2)
    entry_tier2 = round(ma20, 2)
    entry_tier3 = round(min(ma60, recent_low_20), 2)
    stop_loss_price = round(min(ma60 * 0.96, recent_low_60 * 0.98), 2)
    stop_loss_pct = round(((stop_loss_price - curr_price) / curr_price) * 100, 2)
    tp_tier1 = round(target_mean, 2) if has_target else round(curr_price * 1.15, 2)
    tp_tier2 = round(max(target_high if has_target else 0, recent_high_60 * 1.1), 2)

    # 5. 教練建議
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

    # 目標價區塊格式化
    if has_target:
        analyst_section = f"• 涵蓋機構：{analyst_count} 家法人報告\n• 平均目標：{target_mean:.2f} 元 (空間: {target_upside:+.1f}%)\n• 目標區間：{target_low:.2f} ~ {target_high:.2f} 元\n"
        if target_details:
            analyst_section += "• 近期法人評等：\n"
            for t in target_details:
                analyst_section += f"  - {t['broker']}: {t['rating']} (目標價 {t['target']} 元)\n"
    else:
        analyst_section = "• 暫無近半年公開目標價（中小型冷門股或未獲大型投顧出具報告）。\n"

    report = (
        f"📊【{company_name} ({code}) 深度診斷報告】\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"📌 現價：{curr_price:.2f} 元 | 月線：{ma20:.2f} | 季線：{ma60:.2f}\n\n"
        f"一、🎯 法人目標價與評等共識\n"
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
