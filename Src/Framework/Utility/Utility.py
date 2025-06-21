import os
import datetime

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