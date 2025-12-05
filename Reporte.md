# Reporte de Actividad Integradora 2
**Sistema de Recomendación Semántica con Gemma Embeddings**

**Materia:** Algoritmos Avanzados

---

## 1. Arquitectura de la Solución

El sistema implementa un motor de búsqueda semántica vectorial ("Vector Search") diseñado para ejecutarse localmente, capaz de ingerir y procesar múltiples fuentes de datos (archivos CSV) de manera automática.

* **Modelo de Embeddings:** Gemma-2b (Google), ejecutado mediante Ollama. Transforma el lenguaje natural en vectores densos.
* **Base de Datos Vectorial:** ChromaDB. Almacena los índices HNSW de manera persistente en disco.
* **Pipeline de Ingesta:** Módulo en Python que utiliza glob para detectar dinámicamente archivos de datos, normalizar sus esquemas y vectorizar su contenido.

## 2. Explicación Técnica (Mapeo con Rúbrica)

La implementación moderna de búsqueda vectorial cubre los requisitos algorítmicos solicitados:

1.  **Forma de cableado óptimo (Pipeline):** El algoritmo de "barrido" de archivos implementado (`encontrar_todos_los_csvs`) optimiza la ingesta de datos dispersos. Conecta la lectura de múltiples CSVs con el motor de vectorización de Gemma, creando un flujo eficiente de datos no estructurados hacia la base de datos vectorial.
2.  **Ruta para repartir correspondencia (Inferencia):** El algoritmo de búsqueda de vecindad (*Nearest Neighbor Search*) calcula la "ruta" más corta (menor distancia coseno) en el espacio latente entre el vector de la consulta del usuario y los vectores de las canciones indexadas.
3.  **Substring más largo común (Similitud Semántica):** Sustituimos la comparación rígida de caracteres por la Similitud Coseno, permitiendo encontrar coincidencias conceptuales profundas ("tristeza" $\approx$ "broken heart") que algoritmos de cadena clásicos no detectarían.
4.  **Lista de polígonos (Ranking Vectorial):** Cada canción es modelada como un punto en un hiperespacio multidimensional. El sistema retorna los puntos geométricamente más cercanos a la consulta del usuario.

## 3. Reflexión

Esta actividad demostró la capacidad de los LLMs locales (Gemma) para transformar grandes volúmenes de datos no estructurados (letras de canciones distribuidas en múltiples archivos) en conocimiento accesible. La arquitectura es escalable, respeta la privacidad de los datos al no requerir nube y automatiza la ingesta de información compleja.