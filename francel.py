import cv2
import os
import numpy as np

def calcular_nitidez(imagen):
    """Calcula la varianza del Laplaciano para medir el enfoque (nitidez)."""
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gris, cv2.CV_64F).var()

def extraer_fotogramas_optimizados(ruta_video, carpeta_salida, fps_deseados=1):
    # Crear carpeta de salida si no existe
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)
        print(f"Carpeta creada: {carpeta_salida}")

    # Capturar el video
    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        print("Error: No se pudo abrir el archivo de video.")
        return

    fps_video = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calcular cuántos fotogramas componen el bloque de tiempo (ej. 1 segundo)
    intervalo_cuadros = int(fps_video / fps_deseados)
    if intervalo_cuadros < 1:
        intervalo_cuadros = 1

    print(f"FPS del Video Original: {fps_video:.2f}")
    print(f"Total de cuadros en video original: {total_frames}")
    print(f"Estrategia de reducción activa: Seleccionando el MEJOR cuadro por cada segundo...")

    contador_guardados = 0
    cuadro_actual = 0

    while cap.isOpened():
        fotos_bloque = []
        nitideces = []
        
        # Agrupamos los cuadros del intervalo de 1 segundo para elegir el de mejor calidad
        for _ in range(intervalo_cuadros):
            ret, frame = cap.read()
            if not ret:
                break
            cuadro_actual += 1
            
            # Almacenamos temporalmente el cuadro y su índice de nitidez
            fotos_bloque.append(frame)
            nitideces.append(calcular_nitidez(frame))

        # Si el bloque está vacío porque el video terminó, salimos del bucle
        if not fotos_bloque:
            break

        # Seleccionamos el índice del cuadro que tuvo la mayor varianza (mayor nitidez)
        mejor_indice = np.argmax(nitideces)
        mejor_cuadro = fotos_bloque[mejor_indice]
        
        # Guardar el cuadro seleccionado
        contador_guardados += 1
        nombre_archivo = os.path.join(carpeta_salida, f"frame_{contador_guardados:04d}.jpg")
        
        # [cv2.IMWRITE_JPEG_QUALITY, 100] fuerza la máxima fidelidad de compresión para el gemelo digital
        cv2.imwrite(nombre_archivo, mejor_cuadro, [cv2.IMWRITE_JPEG_QUALITY, 100])
        
        # Mostrar el progreso en la terminal
        porcentaje = (cuadro_actual / total_frames) * 100
        print(f"Progreso: {porcentaje:.1f}% | Analizados: {cuadro_actual}/{total_frames} | Guardado: frame_{contador_guardados:04d}.jpg", end="\r")

    cap.release()
    print(f"\n\n¡Proceso de reducción finalizado con éxito!")
    print(f"Se redujo el video de 5 minutos a solo {contador_guardados} fotogramas de alta nitidez.")
    print(f"Imágenes guardadas en la carpeta: '{carpeta_salida}'")

# --- CONFIGURACIÓN DE PARÁMETROS ---
if __name__ == "__main__":
    # 1. Pon aquí el nombre exacto de tu archivo de video de 5 minutos
    RUTA_DEL_VIDEO = "mi_video.mp4" 
    
    # 2. Nombre de la carpeta limpia que se generará
    CARPETA_DESTINO = "fotogramas_reducidos"
    
    # 3. Cuántos cuadros guardar por segundo. 
    # Al dejarlo en 1, procesará bloques de 1 segundo entero y extraerá la foto más limpia de ese lapso.
    FPS_EXTRACCION = 1

    extraer_fotogramas_optimizados(RUTA_DEL_VIDEO, CARPETA_DESTINO, fps_deseados=FPS_EXTRACCION)