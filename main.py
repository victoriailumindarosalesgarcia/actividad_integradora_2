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

class SistemaRecomendacion:
    def __init__(self):
        print(f"\n🤖 Inicializando sistema con {MODELO_GEMMA}...")
        self.directorio_actual = os.path.dirname(os.path.abspath(__file__))
        
        # Guardamos la DB en una carpeta local para que sea persistente
        self.ruta_db = os.path.join(self.directorio_actual, "chroma_db")

        try:
            self.embeddings = OllamaEmbeddings(model=MODELO_GEMMA)
            self.chroma_client = chromadb.PersistentClient(path=self.ruta_db)
            self.collection = self.chroma_client.get_or_create_collection(
                name=NOMBRE_COLECCION,
                metadata={"hnsw:space": "cosine"}
            )
            print("✅ Motor vectorial listo.")
        except Exception as e:
            print(f"❌ Error al iniciar motor: {e}")
            sys.exit(1)

    def encontrar_csv(self):
        """Busca automáticamente cualquier archivo .csv en la carpeta actual"""
        print("🔍 Buscando archivos CSV en la carpeta...")
        archivos = glob.glob(os.path.join(self.directorio_actual, "*.csv"))
        
        if archivos:
            archivo_encontrado = archivos[0]
            print(f"📂 Archivo encontrado: {os.path.basename(archivo_encontrado)}")
            return archivo_encontrado
        else:
            print("❌ No encontré ningún archivo .csv en esta carpeta.")
            return None

    def generar_dataset_emergencia(self):
        """Crea un dataset pequeño si no encuentra el ZIP descomprimido"""
        print("⚠️ Generando datos de prueba de emergencia...")
        data = {
            "artist": ["Coldplay", "Adele", "Survivor", "Queen"],
            "title": ["Fix You", "Someone Like You", "Eye of the Tiger", "Don't Stop Me Now"],
            "lyrics": [
                "Lights will guide you home and ignite your bones",
                "Never mind I'll find someone like you",
                "It's the eye of the tiger it's the thrill of the fight",
                "Tonight I'm gonna have myself a real good time"
            ]
        }
        df = pd.DataFrame(data)
        ruta = os.path.join(self.directorio_actual, "dataset_emergencia.csv")
        df.to_csv(ruta, index=False)
        return ruta

    def indexar_datos(self):
        # 1. Buscar CSV real
        ruta_csv = self.encontrar_csv()
        
        # 2. Si no hay, crear uno de emergencia para que puedas entregar
        if not ruta_csv:
            ruta_csv = self.generar_dataset_emergencia()

        try:
            # Leemos el CSV
            df = pd.read_csv(ruta_csv)
            
            # Limpieza de nombres de columnas (por si vienen en mayúsculas o distinto)
            df.columns = [c.lower().strip() for c in df.columns]
            
            # Intentamos adivinar las columnas correctas
            col_titulo = next((c for c in df.columns if 'title' in c or 'song' in c or 'name' in c), None)
            col_artista = next((c for c in df.columns if 'artist' in c), None)
            col_letra = next((c for c in df.columns if 'lyric' in c or 'text' in c), None)

            if not (col_titulo and col_artista and col_letra):
                print("❌ No pude identificar las columnas de Título, Artista o Letra en el CSV.")
                print(f"Columnas encontradas: {df.columns}")
                return

            # IMPORTANTE: Tomamos solo las primeras 50 canciones para que sea rápido
            # Procesar 50,000 canciones en local tardaría horas.
            df_mini = df.head(50)
            print(f"⚡ Procesando las primeras {len(df_mini)} canciones para la demostración...")
            
            ids = []
            docs = []
            metas = []

            start = time.time()
            print("⏳ Vectorizando con Gemma (esto puede tardar unos segundos)...")

            for idx, row in df_mini.iterrows():
                try:
                    # Convertimos a string y limpiamos
                    t = str(row[col_titulo])
                    a = str(row[col_artista])
                    l = str(row[col_letra])[:500] # Recortamos letras gigantes
                    
                    texto_vectorizar = f"Canción: {t}. Artista: {a}. Letra: {l}"
                    
                    ids.append(str(idx))
                    docs.append(texto_vectorizar)
                    metas.append({"titulo": t, "artista": a})
                except:
                    continue
            
            if docs:
                vectores = self.embeddings.embed_documents(docs)
                self.collection.add(ids=ids, embeddings=vectores, documents=docs, metadatas=metas)
                print(f"✅ ¡Listo! Indexación terminada en {time.time() - start:.2f}s")
            else:
                print("⚠️ No se pudieron extraer datos válidos del CSV.")

        except Exception as e:
            print(f"❌ Error procesando el archivo: {e}")

    def buscar(self, consulta):
        print(f"\n🔎 Buscando: '{consulta}'...")
        try:
            vector_query = self.embeddings.embed_query(consulta)
            resultados = self.collection.query(query_embeddings=[vector_query], n_results=3)
            
            print("\n🎶 Recomendaciones:")
            if not resultados['ids'] or not resultados['ids'][0]:
                print("No se encontraron coincidencias.")
                return

            for i in range(len(resultados['ids'][0])):
                titulo = resultados['metadatas'][0][i]['titulo']
                artista = resultados['metadatas'][0][i]['artista']
                score = 1 - resultados['distances'][0][i]
                print(f"{i+1}. {titulo} - {artista} (Similitud: {score:.2f})")
        except Exception as e:
            print(f"❌ Error durante la búsqueda: {e}")

if __name__ == "__main__":
    app = SistemaRecomendacion()
    
    # Si la base de datos está vacía, buscamos CSV e indexamos
    if app.collection.count() == 0:
        app.indexar_datos()
    else:
        print("ℹ️  Base de datos vectorial cargada.")

    while True:
        q = input("\n>> Escribe un sentimiento o situación (o 'salir'): ")
        if q.lower() in ['salir', 'exit']: break
        if q.strip(): app.buscar(q)