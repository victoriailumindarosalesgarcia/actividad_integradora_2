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
# Límite de seguridad: Procesamos máximo 100 canciones para que la demo sea rápida.
# Si quieres procesar todo, cambia esto a 10000 o None (pero tardará horas).
LIMITE_CANCIONES = 100 

class SistemaRecomendacion:
    def __init__(self):
        print(f"\n🤖 Inicializando sistema con {MODELO_GEMMA}...")
        self.dir_actual = os.path.dirname(os.path.abspath(__file__))
        self.ruta_db = os.path.join(self.dir_actual, "chroma_db")

        try:
            self.embeddings = OllamaEmbeddings(model=MODELO_GEMMA)
            self.chroma_client = chromadb.PersistentClient(path=self.ruta_db)
            self.collection = self.chroma_client.get_or_create_collection(
                name=NOMBRE_COLECCION,
                metadata={"hnsw:space": "cosine"}
            )
            print("✅ Motor vectorial listo.")
        except Exception as e:
            print(f"❌ Error crítico inicializando: {e}")
            sys.exit(1)

    def encontrar_todos_los_csvs(self):
        """Busca TODOS los archivos .csv en la carpeta actual"""
        patron = os.path.join(self.dir_actual, "*.csv")
        archivos = glob.glob(patron)
        print(f"🔍 Buscando CSVs en: {self.dir_actual}")
        print(f"📂 Archivos encontrados: {len(archivos)}")
        return archivos

    def indexar_datos(self):
        csvs = self.encontrar_todos_los_csvs()
        
        if not csvs:
            print("❌ ERROR: No encontré ningún archivo .csv en esta carpeta.")
            print("   Asegúrate de descomprimir el ZIP aquí mismo.")
            return

        total_procesadas = 0
        ids = []
        docs = []
        metas = []

        print(f"⚡ Comenzando indexación masiva (Límite: {LIMITE_CANCIONES} canciones)...")
        start_total = time.time()

        for archivo in csvs:
            if total_procesadas >= LIMITE_CANCIONES:
                break
            
            print(f"   📄 Leyendo: {os.path.basename(archivo)}...")
            try:
                # Leemos el CSV (manejando errores de formato comunes)
                df = pd.read_csv(archivo, on_bad_lines='skip', encoding='utf-8', engine='python')
                
                # Normalizamos columnas
                df.columns = [c.lower().strip() for c in df.columns]
                
                # Detectives de columnas: Buscamos variaciones de nombres
                col_t = next((c for c in df.columns if c in ['title', 'song', 'track_name', 'name']), None)
                col_a = next((c for c in df.columns if c in ['artist', 'singer', 'band', 'performer']), None)
                col_l = next((c for c in df.columns if c in ['lyrics', 'text', 'lyric', 'content']), None)

                # Si no encontramos las columnas, saltamos este archivo
                if not (col_t and col_l):
                    print(f"      ⚠️ Saltando {os.path.basename(archivo)} (No detecté columnas de Título/Letra)")
                    continue

                # Procesamos filas
                for _, row in df.iterrows():
                    if total_procesadas >= LIMITE_CANCIONES:
                        break
                    
                    try:
                        tit = str(row[col_t])
                        # Si no hay columna artista, ponemos "Desconocido"
                        art = str(row[col_a]) if col_a else "Unknown Artist"
                        let = str(row[col_l])[:1000] # Cortamos letras gigantes para velocidad
                        
                        # Solo procesamos si hay letra real
                        if len(let) > 20: 
                            texto_vector = f"Song: {tit}. Artist: {art}. Context: {let}"
                            
                            ids.append(f"song_{total_procesadas}")
                            docs.append(texto_vector)
                            metas.append({"titulo": tit, "artista": art})
                            total_procesadas += 1
                    except:
                        continue

            except Exception as e:
                print(f"      ❌ Error leyendo archivo: {e}")

        if docs:
            print(f"⏳ Vectorizando {len(docs)} canciones con Gemma (paciencia)...")
            vectores = self.embeddings.embed_documents(docs)
            self.collection.add(ids=ids, embeddings=vectores, documents=docs, metadatas=metas)
            print(f"✅ ¡Éxito! Indexación terminada en {time.time() - start_total:.2f}s")
        else:
            print("⚠️ No se pudieron extraer canciones válidas de los CSVs encontrados.")

    def buscar(self, consulta):
        print(f"\n🔎 Buscando: '{consulta}'...")
        try:
            vector_query = self.embeddings.embed_query(consulta)
            resultados = self.collection.query(query_embeddings=[vector_query], n_results=3)
            
            print("\n🎶 RECOMENDACIONES SEMÁNTICAS:")
            print("="*40)
            
            if not resultados['ids'] or not resultados['ids'][0]:
                print("No se encontraron coincidencias.")
                return

            for i in range(len(resultados['ids'][0])):
                titulo = resultados['metadatas'][0][i]['titulo']
                artista = resultados['metadatas'][0][i]['artista']
                score = 1 - resultados['distances'][0][i]
                print(f"{i+1}. {titulo}")
                print(f"   👤 {artista}")
                print(f"   📊 Similitud: {score:.4f}")
                print("-" * 40)
        except Exception as e:
            print(f"❌ Error durante la búsqueda: {e}")

if __name__ == "__main__":
    app = SistemaRecomendacion()
    
    # Si la base de datos está vacía, buscamos CSVs e indexamos
    if app.collection.count() == 0:
        app.indexar_datos()
    else:
        print(f"ℹ️  Base de datos cargada ({app.collection.count()} canciones listas).")

    while True:
        q = input("\n>> Describe una situación o sentimiento (o 'salir'): ")
        if q.lower() in ['salir', 'exit']: break
        if q.strip(): app.buscar(q)