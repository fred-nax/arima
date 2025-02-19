import streamlit as st
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Titre de l'application
st.title("📈 Prédiction des Ventes avec ARIMA")

# Upload du fichier CSV
uploaded_file = st.file_uploader("📂 Chargez un fichier CSV contenant les ventes", type=["csv"])

if uploaded_file:
    # Charger les données
    df = pd.read_csv(uploaded_file)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")

    # Vérifier si la colonne 'Ventes' est bien présente
    if "Ventes" not in df.columns:
        st.error("Le fichier doit contenir une colonne nommée 'Ventes'")
    else:
        # Afficher les premières lignes
        st.write("🔍 Aperçu des données :")
        st.write(df.head())

        # Entraîner ARIMA
        model = ARIMA(df["Ventes"], order=(1,1,1))
        model_fit = model.fit()

        # Faire des prévisions pour les 6 prochains mois
        future_steps = 6
        forecast = model_fit.forecast(steps=future_steps)
        future_dates = pd.date_range(start=df.index[-1], periods=future_steps+1, freq="M")[1:]
        df_forecast = pd.DataFrame({"Date": future_dates, "Prévision_Ventes": forecast.values})

        # Afficher le graphique
        st.subheader("📊 Graphique des Prévisions")
        plt.figure(figsize=(10,5))
        plt.plot(df.index, df["Ventes"], marker="o", linestyle="-", label="Ventes réelles")
        plt.plot(df_forecast["Date"], df_forecast["Prévision_Ventes"], marker="o", linestyle="--", label="Prévisions", color="red")
        plt.xlabel("Date")
        plt.ylabel("Ventes")
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

        # Afficher les prévisions sous forme de tableau
        st.subheader("📅 Prévisions des 6 prochains mois")
        st.write(df_forecast)
