from pathlib import Path
import pandas as pd 
from sklearn.model_selection import train_test_split

raiz = Path(__file__).parent.parent / "dataset_depurado"

registro = []

for documento in raiz.rglob("*"):
    if not documento.is_dir():
        continue

    if any(child.is_dir() for child in documento.iterdir()):
        continue

    partes = documento.parts

    if "cursiva" in partes:
        if "texto" in partes:
            if "con_traspaso" in partes:
                categoria = "texto_con_traspaso"
            elif "sin_traspaso" in partes:
                categoria = "texto_sin_traspaso"
            else:
                categoria = "texto"
        elif "texto_ilustracion" in partes:
            categoria = "texto_ilustracion"
        elif "ilustracion" in partes:
            categoria = "ilustracion"
        else:
            continue
    else:
        continue

    registro.append({
        "documento": documento.name,
        "ruta": str(documento),
        "categoria": categoria
    })

df = pd.DataFrame(registro)

test_val, entrenamiento = train_test_split(
    df,
    test_size=0.7,
    train_size=0.3,
    random_state=42,
    shuffle=True,
    stratify=df["categoria"]
)

#print(entrenamiento["categoria"].value_counts(normalize=True))

test_val.to_csv("dataset_representativo/notas/indice_prueba_evaluacion.csv")
entrenamiento.to_csv("dataset_representativo/notas/indice_entrenamiento.csv")

test_val_filtrado = test_val.query("categoria not in ['ilustracion', 'texto_ilustracion']")

test, val = train_test_split(
    test_val_filtrado,
    test_size=0.5,
    train_size=0.5,
    random_state=42,
    shuffle=True,
    stratify=test_val_filtrado["categoria"]
)

test.to_csv("dataset_representativo/notas/indice_prueba.csv")
val.to_csv("dataset_representativo/notas/indice_evaluacion.csv")