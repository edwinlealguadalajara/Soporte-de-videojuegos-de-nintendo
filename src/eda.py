import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def crear_directorios():
    os.makedirs("data", exist_ok=True)
    os.makedirs("reports", exist_ok=True)


def parsear_fecha_vg(fecha_str):
    if pd.isna(fecha_str):
        return pd.NaT
    limpia = (
        pd.Series([str(fecha_str)])
        .str.replace(r"(\d+)(st|nd|rd|th)", r"\1", regex=True)
        .iloc[0]
    )
    return pd.to_datetime(limpia, format="%d %b %y", errors="coerce")


def cargar_datos():
    archivos = [
        f for f in glob.glob("data/*.csv")
        if not f.endswith(("limpio.csv", "test.csv"))
    ]
    if not archivos:
        raise FileNotFoundError("No se encontró ningún CSV original dentro de la carpeta data/.")
    
    ruta_csv = archivos[0]
    print(f"Cargando dataset desde: {ruta_csv}\n")
    df = pd.read_csv(ruta_csv, encoding="latin-1")
    print("=== PRIMERAS 5 FILAS DEL DATASET NINTENDO SWITCH ===")
    print(df[["Position", "Game", "Publisher", "Release Date", "Last Update"]].head(), "\n")
    return df


def analisis_descriptivo(df):
    print("=== INFORMACIÓN GENERAL ===")
    print(f"Dimensiones originales: {df.shape[0]} filas x {df.shape[1]} columnas\n")
    print("=== VALORES NULOS POR COLUMNA ===")
    print(df.isnull().sum(), "\n")
    print("=== ESTADÍSTICAS DESCRIPTIVAS (NUMÉRICAS ORIGINALES) ===")
    print(df.describe().T[["count", "mean", "std", "min", "50%", "max"]], "\n")


def limpiar_y_preparar_datos(df):
    df_clean = df.copy()

    # 1. Limpiar saltos de línea en nombres de juegos
    df_clean["Game"] = df_clean["Game"].str.replace(r"\s+", " ", regex=True).str.strip()

    # 2. Parsear fechas y calcular días de mantenimiento post-lanzamiento
    df_clean["Release_dt"] = df_clean["Release Date"].apply(parsear_fecha_vg)
    df_clean["Update_dt"] = df_clean["Last Update"].apply(parsear_fecha_vg)
    df_clean = df_clean.dropna(subset=["Release_dt", "Update_dt"]).copy()

    df_clean["Dias_Mantenimiento"] = (
        df_clean["Update_dt"] - df_clean["Release_dt"]
    ).dt.days

    # Conservar juegos con soporte post-lanzamiento (>= 0 días)
    df_clean = df_clean[df_clean["Dias_Mantenimiento"] >= 0].copy()

    # 3. Variable objetivo binaria usando la mediana (108 días)
    # 0 = Soporte Corto (<= 108 días) | 1 = Soporte Prolongado (> 108 días)
    mediana_dias = df_clean["Dias_Mantenimiento"].median()
    df_clean["Target_Mantenimiento"] = (
        df_clean["Dias_Mantenimiento"] > mediana_dias
    ).astype(int)

    # 4. Transformar Total Shipped ("45.33m" -> 45.33)
    df_clean["Total_Shipped_M"] = (
        df_clean["Total Shipped"].str.replace("m", "", regex=False).astype(float)
    )
    df_clean["Tiene_Ventas_Reportadas"] = df_clean["Total_Shipped_M"].notna().astype(int)
    df_clean["Total_Shipped_M"] = df_clean["Total_Shipped_M"].fillna(0.0)

    # 5. Banderas de disponibilidad e imputación con mediana para puntajes
    for col in ["VGChartz Score", "Critic Score", "User Score"]:
        col_flag = f"Tiene_{col.replace(' ', '_')}"
        df_clean[col_flag] = df_clean[col].notna().astype(int)
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # 6. Extraer variables temporales y codificar Publisher / Developer
    df_clean["Release_Year"] = df_clean["Release_dt"].dt.year
    df_clean["Release_Month"] = df_clean["Release_dt"].dt.month
    df_clean["Release_DayOfWeek"] = df_clean["Release_dt"].dt.dayofweek
    df_clean["Es_Nintendo"] = (df_clean["Publisher"].str.strip() == "Nintendo").astype(int)

    df_clean["Publisher_Freq"] = df_clean["Publisher"].map(
        df_clean["Publisher"].value_counts()
    )
    df_clean["Developer_Freq"] = (
        df_clean["Developer"].map(df_clean["Developer"].value_counts()).fillna(0)
    )

    columnas_finales = [
        "Position",
        "VGChartz Score",
        "Critic Score",
        "User Score",
        "Total_Shipped_M",
        "Tiene_Ventas_Reportadas",
        "Tiene_VGChartz_Score",
        "Tiene_Critic_Score",
        "Tiene_User_Score",
        "Release_Year",
        "Release_Month",
        "Release_DayOfWeek",
        "Es_Nintendo",
        "Publisher_Freq",
        "Developer_Freq",
        "Dias_Mantenimiento",
        "Target_Mantenimiento",
    ]

    df_modelo = df_clean[columnas_finales].drop_duplicates().reset_index(drop=True)

    print(f"=== RESUMEN DEL TARGET (Mediana = {mediana_dias:.0f} días) ===")
    print(df_modelo["Target_Mantenimiento"].value_counts(), "\n")

    df_modelo.to_csv("data/switch_limpio.csv", index=False)
    print(f"Dataset limpio guardado en 'data/switch_limpio.csv' ({df_modelo.shape})\n")
    return df_modelo


def visualizar_distribuciones(df_modelo):
    # 1. Histogramas
    cols_hist = [
        "Position",
        "Dias_Mantenimiento",
        "Release_Year",
        "Release_Month",
        "Publisher_Freq",
        "Target_Mantenimiento",
    ]
    df_modelo[cols_hist].hist(figsize=(12, 8), bins=20, color="#e60012", edgecolor="black")
    plt.suptitle("Distribución de Variables - Nintendo Switch", fontsize=14)
    plt.tight_layout()
    plt.savefig("reports/histogramas.png", dpi=150)
    plt.close()

    # 2. Boxplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(
        data=df_modelo,
        x="Es_Nintendo",
        y="Dias_Mantenimiento",
        ax=axes[0],
        palette="Set2",
    )
    axes[0].set_title("Días de Mantenimiento: Terceros (0) vs Nintendo (1)")
    sns.boxplot(
        data=df_modelo,
        x="Target_Mantenimiento",
        y="Position",
        ax=axes[1],
        palette="coolwarm",
    )
    axes[1].set_title("Posición en Ranking por Clase de Mantenimiento")
    plt.tight_layout()
    plt.savefig("reports/boxplots.png", dpi=150)
    plt.close()

    # 3. Matriz de correlación
    plt.figure(figsize=(11, 8))
    cols_corr = [
        "Target_Mantenimiento",
        "Position",
        "Total_Shipped_M",
        "Critic Score",
        "User Score",
        "Release_Year",
        "Es_Nintendo",
        "Publisher_Freq",
        "Developer_Freq",
    ]
    corr = df_modelo[cols_corr].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Matriz de Correlación - Nintendo Switch")
    plt.tight_layout()
    plt.savefig("reports/matriz_correlacion.png", dpi=150)
    plt.close()

    print("Gráficas generadas correctamente en la carpeta 'reports/'.")


if __name__ == "__main__":
    crear_directorios()
    df_raw = cargar_datos()
    analisis_descriptivo(df_raw)
    df_limpio = limpiar_y_preparar_datos(df_raw)
    visualizar_distribuciones(df_limpio)
