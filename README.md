# TF-IDF CLI en C++

Aplicación por línea de comandos que procesa un conjunto de ficheros `.txt`, aplica filtrado por stop words y lematización, y calcula TF, IDF, TF-IDF y la matriz de similitud coseno entre documentos.

Requisitos
- g++ con soporte C++17

Cómo compilar (el ejecutable se genera en /bin por cómo está configurado el Makefile)

```bash
make
```

Uso

- Usando un directorio de documentos:

```bash
./bin/tfidf -d ejemplos -s modelos/stopwords.txt -l modelos/lemmas.txt
```

- Pasando archivos concretos:

```bash
./bin/tfidf -s modelos/stopwords.txt -l modelos/lemmas.txt ejemplos/document-01.txt ejemplos/document-02.txt
```

Salida
- Para cada documento imprime una tabla con columnas: Índice del término, Término, TF, IDF, TF-IDF.
- Después imprime la matriz de similitud coseno entre documentos (valores en [0,1]).

Formato de ficheros auxiliares
- `stopwords.txt`: lista de palabras (una por línea) que se eliminarán del procesamiento.
- `lemmas.txt`: mapeo de token a lema, cada línea: `forma lema` (separados por espacio). Ambos serán convertidos a minúsculas.

Notas
- TF se calcula como recuento bruto de apariciones.
- IDF se calcula como log(N / df) donde N es el número de documentos y df es en cuántos documentos aparece el término.

Ejemplos incluidos en `ejemplos/`.
