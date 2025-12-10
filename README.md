# ============================================
 # 1) Calling the libraries 
# ============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ============================================
 # 2) Data entry (gold prices for November)
# ============================================
prices = [
    174850, 175100, 175300, 175500, 175700, 175950,
    176100, 176200, 176300, 176400, 176500, 176600,
    176700, 176800, 176850, 176900, 176950, 177000,
    177050, 177100, 177120, 177140, 177160, 177180,
    177190, 177200, 177210, 177215, 177220, 177224
]

df = pd.DataFrame({
    "Day": np.arange(1, 31),
    "Price": prices
})

# ============================================
 # 3) Data preparation for the model (Regression)
# ============================================
X = df[["Day"]]     
y = df["Price"]     

model = LinearRegression()
model.fit(X, y)

# ============================================
# 4) Tomorrow's price prediction
# ============================================
next_day = np.array([[31]])
predicted_price = model.predict(next_day)[0]
current_price = df["Price"].iloc[-1]

print("Current gold price (per gram):", f"{current_price:,.0f} IQD")
print("Predicted gold price for tomorrow (per gram):", f"{predicted_price:,.2f} IQD")

# ============================================
# 5) graph
# ============================================
plt.plot(df["Day"], df["Price"], marker='o')
plt.axhline(y=predicted_price, linestyle='--')

advice_text = " Do not invest today" if predicted_price < current_price else " Investment may be good"
color = "red" if predicted_price < current_price else "green"

plt.text(30.5, predicted_price, advice_text, color=color)

plt.xlabel("Day")
plt.ylabel("Gold Price (IQD)")
plt.title("Gold Price Prediction")
plt.show()

# ============================================
# 6)  Printing
# ============================================
if predicted_price < current_price:
    print(" Advice: The predicted price tomorrow is lower than today, better not to invest today.")
else:
    print(" Advice: The predicted price tomorrow is higher than today, investment may be a good option.")

    grams = float(input("Enter the number of grams you want to invest in: "))
    profit_per_gram = predicted_price - current_price
    total_profit = profit_per_gram * grams
    total_value = (current_price * grams) + total_profit

    print("Profit per gram:", f"{profit_per_gram:,.2f} IQD")
    print("Expected profit from your investment:", f"{total_profit:,.2f} IQD")
    print("Total investment value tomorrow:", f"{total_value:,.2f} IQD")
