import cv2
import numpy as np

def verificar_quadrado(imagem_path):

    imagem = cv2.imread(imagem_path)
    if imagem is None:
        return "Erro ao carregar imagem"

    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('./image/imagem_cinza.jpg', imagem_cinza) 

    # Aplica um filtro de bordas (Canny)
    bordas = cv2.Canny(imagem_cinza, 100, 200)

    # Encontra os contornos na imagem
    contornos, _ = cv2.findContours(bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Itera sobre os contornos encontrados
    for contorno in contornos:
        # Aproxima o contorno para uma forma poligonal
        epsilon = 0.04 * cv2.arcLength(contorno, True)
        contorno_aproximado = cv2.approxPolyDP(contorno, epsilon, True)

        # Verifica se o contorno tem 4 vértices (quadrado ou retângulo)
        if len(contorno_aproximado) == 4:
            # Verifica se os ângulos são próximos de 90 graus (quadrado)
            # Para um quadrado perfeito, a relação de lados seria 1:1
            # (relacionamento entre largura e altura)
            x, y, w, h = cv2.boundingRect(contorno_aproximado)
            if abs(w - h) < 0.1 * max(w, h):  # A diferença entre largura e altura deve ser pequena
                return "Sim, tem um quadrado na imagem"
    
    # Se nenhum quadrado for encontrado
    return "Não, não há um quadrado na imagem"

# Teste da função
imagem_path = 'test_4.png'
resultado = verificar_quadrado(imagem_path)
print(resultado)
