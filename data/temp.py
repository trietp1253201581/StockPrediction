import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Lấy dữ liệu 5 năm của một cổ phiếu
ticker = "AAPL"
df = yf.download(ticker, start="2019-03-01", end="2024-03-01")

# Tính SMA 50 và SMA 200
df["SMA_50"] = df["Close"].rolling(50).mean()
df["SMA_200"] = df["Close"].rolling(200).mean()

# Tính biến động 30 ngày
df["Volatility"] = df["Close"].pct_change().rolling(30).std()

# Vẽ Line plot xu hướng giá
plt.figure(figsize=(12, 5))
plt.plot(df["Close"], label="Giá đóng cửa")
plt.plot(df["SMA_50"], label="SMA 50 ngày")
plt.plot(df["SMA_200"], label="SMA 200 ngày")
plt.legend()
plt.title(f"Xu hướng giá của {ticker} (5 năm)")
plt.show()

# Vẽ Rolling Volatility
plt.figure(figsize=(12, 5))
plt.plot(df["Volatility"], label="30-day Rolling Volatility")
plt.legend()
plt.title(f"Biến động giá của {ticker} (5 năm)")
plt.show()

# Vẽ Volume Bar Chart theo năm
df["Year"] = df.index.year
df.groupby("Year")["Volume"].mean().plot(kind="bar", figsize=(12, 5), title="Thanh khoản theo năm")
plt.show()

# Vẽ Scatter Plot (Price vs Volume)
plt.figure(figsize=(12, 5))
plt.scatter(df["Close"], df["Volume"], alpha=0.5)
plt.xlabel("Giá đóng cửa")
plt.ylabel("Volume")
plt.title(f"Quan hệ giữa giá và khối lượng giao dịch của {ticker}")
plt.show()
