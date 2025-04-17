# deteccion_baches
# Código utilizado en proyecto en grupo de la asignatura "Laboratorio de proyectos" en el grado de Telecomunicaciones de la Universidade de Vigo.

import os
from tkinter import image_names
import cv2  
import numpy as np

print(cv2.__version__)


def cargar_imagen(folder, image_name):

    print("Cargando imagen...")

    image_path = os.path.join(folder, image_name)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: No se pudo cargar la imagen en {image_path}")
        exit()
    else:
        cv2.imshow("Imagen original", image)  # Mostrar la imagen

    print(
        f"Tamaño de la imagen: {image.shape[1]}x{image.shape[0]} píxeles"
    )  

    gray = cv2.cvtColor(
        image, cv2.COLOR_BGR2GRAY
    )  #   para convertir la imagen a escala de grises

    cv2.imshow("escala grises", gray)
    return image, image_name, gray


def cargar_logo(logo_path, image):

    logo = cv2.imread(logo_path)

    if logo is None:
        print(f"Error: No se pudo cargar el logo en {logo_path}")
        exit()
    
    
    # Redimensionar el logo
    logo_height, logo_width = logo.shape[:2]
    image_height, image_width = image.shape[:2]

    scale_factor = 0.1
    new_width = int(image_width * scale_factor)
    new_height = int(logo_height * (new_width / logo_width))
    logo_resized = cv2.resize(logo, (new_width, new_height))

    # Obtener coordenadas de la esquina inferior derecha
    x_offset = image_width - new_width - 10  # píxeles de margen desde el borde derecho
    y_offset = (
        image_height - new_height - 10
    )  # píxeles de margen desde el borde inferior

    return logo_resized, x_offset, y_offset, new_width, new_height



def corregir_y_ecualizar(gray):
    
    # SUAVIZO LA ILUMINACION DEL FONDO PQ A VECES SALEN LOS BORDES MAS OSCUROS

    background = cv2.GaussianBlur(gray, (51, 51), 0) # desenfoque gaussiano fuerte para modelar la iluminación general
    corrected = cv2.addWeighted(gray, 1.5, background, -0.5, 0) # resto el fondo de la imagen original para compensar la iluminación desigual

    cv2.imshow("Imagen Corregida 1", corrected)

    # METODO NORMALIZACION + ECUALIZACION
    norm_img = cv2.normalize(corrected, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    cv2.imshow("Normalizada", norm_img)
    
    # Ecualizar el histograma para mejorar el contraste (histograma mas uniforme)
    #equalized = cv2.equalizeHist(norm_img)
    #cv2.imshow("Ecualizada", equalized)
    return norm_img



def umbral(norm_img):
    blur = cv2.GaussianBlur(norm_img, (5, 5), 0) # aplicar filtro gaussiano para reducir ruido  (5x5 o 7x7): 
    cv2.imshow("gaussiana (blur)", blur)
    
    # UMBRAL ADAPTATIVO
    thresh = cv2.adaptiveThreshold(
    blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 151, 20
    )
    cv2.imshow("thresh", thresh)

    return thresh


def contornos(thresh):
    
    # Unir contornos cercanos
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(
    thresh, kernel, iterations=1
    )  # Expandir áreas blancas para fusionar contornos cercanos

    # Mostrar imagen umbralizada
    # cv2.imshow("Umbral", thresh)

    min_area = 3000

    # Encontrar contornos y jerarquía
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return min_area,contours,hierarchy


"""
def is_inside(inner, outer):  # Función para verificar si un contorno está completamente dentro de otro

    x, y, w, h = cv2.boundingRect(outer)
    ix, iy, iw, ih = cv2.boundingRect(inner)
    return x < ix and y < iy and (x + w) > (ix + iw) and (y + h) > (iy + ih)
"""
def is_inside(inner, outer):
    """
    Verifica si el contorno 'inner' está completamente dentro del contorno 'outer'.
    """
    for point in inner:
        if cv2.pointPolygonTest(outer, tuple(map(float, point[0])), False) < 0:
            return False  # Si al menos un punto está fuera, el contorno NO está dentro
    return True  # Todos los puntos están dentro


def has_partial_overlap(rect1, rect2): # Función para verificar si un contorno está parcialmente dentro de otro

    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    # Calcular las coordenadas de la intersección
    x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    intersection_area = x_overlap * y_overlap

    # Si hay cualquier solapamiento, los consideramos el mismo bache
    return intersection_area > 0




def seleccion_contornos(image, hierarchy, contours, min_area, is_inside, has_partial_overlap):
    baches = []

    # Filtrar contornos por tamaño mínimo
    for i, contour in enumerate(contours):
        if hierarchy[0][i][3] == -1:  # Solo contornos externos
            area = cv2.contourArea(contour)
            if area > min_area:
                baches.append(contour)


    # Eliminar contronos dentro de otros
    filtered_baches = []
    for i, contour in enumerate(baches):
        is_nested = False  # Asumimos que no está dentro de otro

        for j, other_contour in enumerate(baches):
            if i != j and is_inside(contour, other_contour):   # i distinto de j para no comparar consigo mismos
                print("contorno dentro de otro", j)
                is_nested = True
                break

        if not is_nested:  # Solo agregamos si NO está dentro de otro
            filtered_baches.append(contour)

    # Fusionar contornos solapados
    final_baches = []
    for i, contour in enumerate(filtered_baches):
        rect1 = cv2.boundingRect(contour)
        merged = False

        for j, other_contour in enumerate(final_baches):
            rect2 = cv2.boundingRect(other_contour)

        # Fusionar si se solapan parcialmente
            if has_partial_overlap(rect1, rect2):
                #print("tienen solapado parcial", j)
                final_baches[j] = np.vstack((final_baches[j], contour)) #apila verticalmente en el array
                #print(final_baches[j])
                merged = True
                break

        if not merged:
            final_baches.append(contour)


    # Procesar solo los contornos más grandes
    for contour in final_baches:
        area = cv2.contourArea(contour)
        if area > min_area:
        # Clasificación del bache por tamaño
            if area < 5000:
                category = "Pequeno"
                color = (0, 255, 0)  # Verde
            elif 5000 <= area < 30000:
                category = "Mediano"
                color = (0, 255, 255)  # Amarillo
            else:
                category = "Grande"
                color = (0, 0, 255)  # Rojo

        # Dibujar el contorno y el rectángulo delimitador
            x, y, w, h = cv2.boundingRect(contour)
            #cv2.drawContours(image, [contour], -1, (0,0,0), 2)      # OJO SI DEJAMOS ESTA LINEA SE VEN TB LOS CONTORNOS NEGROS
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

        # Mostrar la clasificación en la imagen
            cv2.putText(
            image,
            category,
            (x + 5, y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        ) # las letras dentro del recuadro del bache


    # Mostrar la imagen con contornos
    cv2.imshow("Contornos Detectados", image)
    return image


# Colocar el logo en la imagen procesada
def imagen_procesada(image, image_name, x_offset, y_offset, new_width, new_height, logo_resized):
    
    image[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = logo_resized
    cv2.imshow("Imagen con logo", image)

    # Crear la carpeta 'imagenes ya procesadas' si no existe
    output_dir = r"C:\Users\lucib\Desktop\lpro\imagenes_ya_procesadas" 
    # os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Carpeta creada: {output_dir}")
        except Exception as e:
            print(f"Error: No se pudo crear la carpeta en {output_dir}. Detalles: {e}")
            exit()
    else:
        print(f"La carpeta ya existe: {output_dir}")

    # Obtener el nombre base de la imagen original (sin la extensión)
    base_name = os.path.splitext(os.path.basename(image_name))[0]

    # Definir la ruta completa donde se guardará la imagen con el nuevo nombre
    output_path = os.path.join(output_dir, base_name + "_procesada.jpg")

    # Guardar la imagen modificada en la carpeta
    cv2.imwrite(output_path, image)
    return image





def main():
       # folder=r"C:\Users\lucib\Desktop\lpro\imagenes_baches_reales_mismo_formato",

    image, image_name, gray = cargar_imagen( folder=r"C:\Users\lucib\Desktop\lpro\imagenes_baches_reales_mismo_formato", image_name="foto_20250325-15_10_09.jpg",
    )

    logo_resized, x_offset, y_offset, new_width, new_height = cargar_logo(
        logo_path=r"C:\Users\lucib\Desktop\lpro\logo.jpg", image=image
    )
    
    norm_img = corregir_y_ecualizar(gray)
    
    thresh = umbral(norm_img)

    min_area, contours, hierarchy = contornos(thresh)
    
    image = seleccion_contornos(image, hierarchy, contours, min_area, is_inside, has_partial_overlap)
    
    imagen_procesada(image, image_name, x_offset, y_offset, new_width, new_height, logo_resized)

    cv2.waitKey(0) # Esperar a que el usuario presione una tecla
    cv2.destroyAllWindows()   # Cerrar todas las ventanas de OpenCV

    
if __name__ == "__main__":
    main()
