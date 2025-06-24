import os
import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class AlertManager:
  def __init__(self, log_path="alerts.log"):
    self.log_path = log_path
    if not os.path.exists(self.log_path):
      with open(self.log_path, "w") as f:
        f.write("=== Alert Log Started ===\n")

  def log_alert(self, message: str):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    with open(self.log_path, "a") as f:
      f.write(log_msg + "\n")

  def check_rsi_alert(self, latest_rsi: float, overbought: float = 70.0, oversold: float = 30.0):
    if latest_rsi >= overbought:
      self.log_alert(f"⚠ RSIが{latest_rsi:.2f}で過熱ゾーン（買われすぎ）に達しています。")
    elif latest_rsi <= oversold:
      self.log_alert(f"⚠ RSIが{latest_rsi:.2f}で売られすぎゾーンに達しています。")

  def check_prediction_alert(self, predicted_close: float, support: float, resistance: float):
    if predicted_close <= support:
      self.log_alert(f"🔻 予測終値がサポートライン({support})を下回る予測: {predicted_close:.2f}")
    elif predicted_close >= resistance:
      self.log_alert(f"🔺 予測終値がレジスタンスライン({resistance})を上回る予測: {predicted_close:.2f}")

class NotificationManager :
  # publicクラスメンバ変数
  loginID  =  ""
  loginPass = ""

  def __init__(self):
    self.loginID   = int(os.getenv('GMAIL_ADDR'))
    self.loginPass = os.getenv('GMAIL_KEY')
    print("[INFO] loginID = "   ,self.loginID)
    print("[INFO] loginPass = " ,self.loginPass)

  def send_email(self, subject, body, to_email):
    msg = MIMEMultipart()
    msg["From"]     = self.loginID
    msg["To"]       = to_email
    msg["Subject"]  = subject
    msg.attach(MIMEText(body, "plain"))

#    try:
#      server = smtplib.SMTP("smtp.gmail.com", 587)
#      server.starttls()
#      server.login(self.loginID, self.loginPass)
#      server.send_message(msg)
#      server.quit()
#      print("[INFO] メール送信成功")
#    except Exception as e:
#      print(f"[ERROR] メール送信失敗: {e}")