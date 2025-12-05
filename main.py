import pandas as pd
import chromadb
import sys
import os
import time
import glob
from langchain_ollama import OllamaEmbeddings

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================
MODELO_GEMMA = "gemma:2b"
NOMBRE_COLECCION = "canciones_gemma_db"

# Límite de seguridad: procesamos máximo 100 canciones para que la demo sea rápida.
# Si quieres procesar todo, cambia esto a 10000 o None (pero tardará mucho más).
LIMITE_CANCIONES = 100

# Número de canciones que devolverá el sistema de recomendación
TOP_K_RESULTADOS = 5


class SistemaRecomendacion:
    def __init__(self) -> None:
        print(f"\n🤖 Inicializando sistema con {MODELO_GEMMA}...")
        self.dir_actual = os.path.dirname(os.path.abspath(__file__))
        self.ruta_db = os.path.join(self.dir_actual, "chroma_db")

        try:
            # Motor de embeddings local con Gemma
            self.embeddings = OllamaEmbeddings(model=MODELO_GEMMA)

            # Cliente persistente de ChromaDB
            self.chroma_client = chromadb.PersistentClient(path=self.ruta_db)

            # Colección (índice vectorial) con métrica coseno
            self.collection = self.chroma_client.get_or_create_collection(
                name=NOMBRE_COLECCION,
                metadata={"hnsw:space": "cosine"},
            )
            print("✅ Motor vectorial listo.")
        except Exception as e:
            print(f"❌ Error crítico inicializando: {e}")
            sys.exit(1)

    # --------------------------------------------------------------------------
    # INDEXACIÓN
    # --------------------------------------------------------------------------
    def encontrar_todos_los_csvs(self):
        """ Busca TODOS los archivos .csv dentro de song_lyrics_dataset/csv. Así nos aseguramos de usar solo el dataset de Kaggle."""
        carpeta_dataset = os.path.join(self.dir_actual, "song_lyrics_dataset", "csv")
        patron = os.path.join(carpeta_dataset, "*.csv")
        archivos = glob.glob(patron)

        print(f"🔍 Buscando CSVs en: {carpeta_dataset}")
        print(f"📂 Archivos .csv encontrados: {len(archivos)}")
        for a in archivos[:5]:
            print(f"   - {os.path.basename(a)}")
        return archivos

    def indexar_datos(self) -> None:
        csvs = self.encontrar_todos_los_csvs()

        if not csvs:
            print("❌ ERROR: No encontré ningún archivo .csv.")
            print("   Asegúrate de descomprimir el ZIP del dataset dentro de este proyecto.")
            return

        total_procesadas = 0
        ids = []
        docs = []
        metas = []

        print(f"⚡ Comenzando indexación masiva (Límite: {LIMITE_CANCIONES} canciones)...")
        start_total = time.time()

        for archivo in csvs:
            if LIMITE_CANCIONES is not None and total_procesadas >= LIMITE_CANCIONES:
                break

            print(f"   📄 Leyendo: {os.path.basename(archivo)}...")
            try:
                # Leemos el CSV manejando errores de formato comunes
                df = pd.read_csv(
                    archivo,
                    on_bad_lines="skip",
                    encoding="utf-8",
                    engine="python",
                )

                # Normalizamos nombres de columnas a minúsculas y sin espacios
                df.columns = [c.lower().strip() for c in df.columns]

                # Detectamos variaciones típicas de columnas
                col_t = next(
                    (c for c in df.columns if c in ["title", "song", "track_name", "name"]),
                    None,
                )
                col_a = next(
                    (c for c in df.columns if c in ["artist", "singer", "band", "performer"]),
                    None,
                )
                col_l = next(
                    (c for c in df.columns if c in ["lyrics", "text", "lyric", "content"]),
                    None,
                )

                # Si no hay título o letra, no nos sirve este archivo
                if not (col_t and col_l):
                    print(
                        f"      ⚠️ Saltando {os.path.basename(archivo)} "
                        "(no detecté columnas de Título/Letra)"
                    )
                    continue

                # Recorremos cada fila del CSV
                for _, row in df.iterrows():
                    if LIMITE_CANCIONES is not None and total_procesadas >= LIMITE_CANCIONES:
                        break

                    try:
                        titulo = str(row[col_t])
                        artista = str(row[col_a]) if col_a else "Unknown Artist"
                        letra = str(row[col_l])

                        # Cortamos letras gigantes para acelerar la demo
                        letra_recortada = letra[:1000]

                        # Solo procesamos si hay letra suficientemente larga
                        if len(letra_recortada) > 20:
                            texto_vector = (
                                f"Song: {titulo}. Artist: {artista}. "
                                f"Context: {letra_recortada}"
                            )

                            ids.append(f"song_{total_procesadas}")
                            docs.append(texto_vector)
                            metas.append({"titulo": titulo, "artista": artista})
                            total_procesadas += 1
                    except Exception:
                        # Si una fila viene mal formateada, simplemente la ignoramos
                        continue

            except Exception as e:
                print(f"      ❌ Error leyendo archivo {os.path.basename(archivo)}: {e}")

        if not docs:
            print("⚠️ No se pudieron extraer canciones válidas de los CSVs encontrados.")
            return

        print(f"⏳ Vectorizando {len(docs)} canciones con Gemma (paciencia)...")
        try:
            vectores = self.embeddings.embed_documents(docs)
            self.collection.add(
                ids=ids,
                embeddings=vectores,
                documents=docs,
                metadatas=metas,
            )
            duracion = time.time() - start_total
            print(f"✅ ¡Éxito! Indexación terminada en {duracion:.2f} segundos.")
        except Exception as e:
            print(f"❌ Error durante la vectorización o el indexado: {e}")

    # --------------------------------------------------------------------------
    # CONSULTA
    # --------------------------------------------------------------------------
    def buscar(self, consulta: str) -> None:
        print(f"\n🔎 Buscando: '{consulta}'...")
        try:
            vector_query = self.embeddings.embed_query(consulta)
            resultados = self.collection.query(
                query_embeddings=[vector_query],
                n_results=TOP_K_RESULTADOS,
            )

            print("\n🎶 RECOMENDACIONES SEMÁNTICAS:")
            print("=" * 40)

            if not resultados.get("ids") or not resultados["ids"][0]:
                print("No se encontraron coincidencias.")
                return

            for i in range(len(resultados["ids"][0])):
                titulo = resultados["metadatas"][0][i]["titulo"]
                artista = resultados["metadatas"][0][i]["artista"]
                # En Chroma, distances es 1 - similitud_coseno cuando el espacio es 'cosine'
                score = 1 - resultados["distances"][0][i]

                print(f"{i + 1}. {titulo}")
                print(f"   👤 {artista}")
                print(f"   📊 Similitud: {score:.4f}")
                print("-" * 40)
        except Exception as e:
            print(f"❌ Error durante la búsqueda: {e}")


# ==============================================================================
# PUNTO DE ENTRADA
# ==============================================================================
if __name__ == "__main__":
    app = SistemaRecomendacion()

    # Si la base está vacía, indexamos; si no, solo avisamos cuántas canciones hay
    try:
        cantidad = app.collection.count()
    except Exception:
        cantidad = 0

    if cantidad == 0:
        app.indexar_datos()
    else:
        print(f"ℹ️ Base de datos cargada ({cantidad} canciones listas).")

    # Bucle interactivo de consulta
    while True:
        q = input("\n>> Describe una situación o sentimiento (o 'salir'): ")
        if q.lower().strip() in {"salir", "exit", "quit"}:
            print("👋 Saliendo del sistema de recomendación.")
            break
        if q.strip():
            app.buscar(q)