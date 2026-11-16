import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def main():
    os.makedirs("reports", exist_ok=True)

    # 1. Cargar el modelo guardado y los datos de prueba
    modelo = joblib.load("models/mejor_modelo.joblib")
    test_df = pd.read_csv("data/switch_test.csv")

    X_test = test_df.drop(columns=["Target_Mantenimiento"])
    y_test = test_df["Target_Mantenimiento"]

    # 2. Evaluar el modelo en datos de prueba
    y_pred = modelo.predict(X_test)

    # 3. Reportar métricas finales
    print("=== MÉTRICAS FINALES EN CONJUNTO DE PRUEBA ===")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1-Score : {f1_score(y_test, y_pred, zero_division=0):.4f}\n")

    print("=== REPORTE DE CLASIFICACIÓN DETALLADO ===")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Soporte Corto (0)", "Soporte Prolongado (1)"]
        )
    )

    # 4. Mostrar y guardar matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    print("=== MATRIZ DE CONFUSIÓN ===")
    print(cm, "\n")

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Reds",
        xticklabels=["Soporte Corto (0)", "Soporte Prolongado (1)"],
        yticklabels=["Soporte Corto (0)", "Soporte Prolongado (1)"]
    )
    plt.title("Matriz de Confusión - Soporte Nintendo Switch")
    plt.xlabel("Predicción del Modelo")
    plt.ylabel("Valor Real")
    plt.tight_layout()
    plt.savefig("reports/matriz_confusion.png", dpi=150)
    plt.close()
    print("Gráfica de matriz de confusión guardada en: reports/matriz_confusion.png")


if __name__ == "__main__":
    main()