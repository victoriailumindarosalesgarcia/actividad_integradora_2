import pandas as pd
import chromadb
import sys
import os
import time
from langchain_ollama import OllamaEmbeddings

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================
# Nombre del modelo en Ollama (debe coincidir con lo que descargaste: 'gemma:2b')
MODELO_GEMMA = "gemma:2b" 
NOMBRE_COLECCION = "canciones_gemma_db"
ARCHIVO_DATASET = "dataset_songs.csv" # Asegúrate que este nombre coincida con el archivo CSV

class SistemaRecomendacion:
    def __init__(self):
        print(f"\n🤖 Inicializando sistema con {MODELO_GEMMA}...")
        
        try:
            # Conexión con Gemma local vía Ollama
            self.embeddings = OllamaEmbeddings(model=MODELO_GEMMA)
            
            # Cliente de base de datos vectorial (persistente para no re-indexar siempre)
            self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
            
            # Colección de vectores
            self.collection = self.chroma_client.get_or_create_collection(
                name=NOMBRE_COLECCION,
                metadata={"hnsw:space": "cosine"}
            )
            print("✅ Motor vectorial listo.")
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)

    def indexar_datos(self):
        """Lee el CSV y crea los vectores con Gemma"""
        if not os.path.exists(ARCHIVO_DATASET):
            print(f"❌ No se encuentra {ARCHIVO_DATASET}")
            return

        df = pd.read_csv(ARCHIVO_DATASET)
        print(f"📂 Cargando {len(df)} canciones...")
        
        ids = []
        docs = []
        metas = []

        print(f"⚡ Vectorizando con Gemma (esto puede tardar unos segundos)...")
        start = time.time()

        for idx, row in df.iterrows():
            # Texto enriquecido para el embedding
            texto = f"Canción: {row['title']}. Artista: {row['artist']}. Letra: {row['lyrics']}"
            
            ids.append(str(idx))
            docs.append(texto)
            metas.append({"titulo": row["title"], "artista": row["artist"]})

        # Generar embeddings
        vectores = self.embeddings.embed_documents(docs)
        
        # Guardar en ChromaDB
        self.collection.add(ids=ids, embeddings=vectores, documents=docs, metadatas=metas)
        
        print(f"✅ Indexación terminada en {time.time() - start:.2f}s")

    def buscar(self, consulta):
        """Busca canciones semánticamente similares"""
        print(f"\n🔎 Buscando: '{consulta}'...")
        
        # Vectorizar consulta
        vector_query = self.embeddings.embed_query(consulta)
        
        # Buscar vecinos cercanos
        resultados = self.collection.query(query_embeddings=[vector_query], n_results=3)
        
        print("\n🎶 Recomendaciones:")
        if not resultados['ids'][0]:
            print("No se encontraron coincidencias.")
            return

        for i in range(len(resultados['ids'][0])):
            titulo = resultados['metadatas'][0][i]['titulo']
            artista = resultados['metadatas'][0][i]['artista']
            score = 1 - resultados['distances'][0][i]
            print(f"{i+1}. {titulo} - {artista} (Similitud: {score:.2f})")

if __name__ == "__main__":
    app = SistemaRecomendacion()
    
    # Si la base de datos está vacía, indexamos
    if app.collection.count() == 0:
        app.indexar_datos()
    else:
        print("ℹ️  Base de datos cargada.")

    while True:
        q = input("\n>> Escribe un sentimiento o situación (o 'salir'): ")
        if q.lower() in ['salir', 'exit']: break
        if q.strip(): app.buscar(q)