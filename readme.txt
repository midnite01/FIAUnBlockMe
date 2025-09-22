Proyecto: Fundamentos de Inteligencia Artificial – Comparación de Algoritmos en "Unblock Me"

---
Integrantes:

-Sergio Villegas Osores integrante 1 (8061)

-Elías Gabriel Pérez Uribe integrante 2 (8061)

---
Nombre del juego: "Unblock Me"

Unblock Me es un rompecabezas de tablero 6x6 (o más) en el que el objetivo es despejarle el camino hasta la salida al bloque rojo, cuando este llega a la salida, ganas el juego. 
Este juego tiene reglas muy simples (que se detallaran a continuación), lo que hace que la dificultad este en "saber que mover".  

Link acerca del juego propuesto: https://play.google.com/store/apps/details?id=com.kiragames.unblockmefree&hl=es_CL

Reglas:
- El bloque rojo se desplaza solo de manera horizontal.
- Otros bloques pueden estar repartidos por el tablero en posicion horizontales o verticales.
- Los bloques solo pueden desplazarse en la direccion paralela a su orientación.
- Los bloques no pueden superponerse ni "atravesarse" entre si.
- Los bloques se mueven dentro de los límites de la cuadrícula.
- El objetivo se alcanza cuando el bloque rojo llega a la salida.

---
Heuristica implementada:

BFS (Breadth First Search):
Este algoritmo no utiliza heurística. Su estrategia consiste en expandir los estados por capas, explorando primero todos los que están a distancia 1, luego a distancia 2, y así sucesivamente desde el estado inicial. 
Eso asegura que encuentre la solución con el menor número de movimientos, pero como no cuenta con ninguna estimación de la cercanía a la meta, tiende a revisar muchos más nodos que un algoritmo informado.

A* (A estrella):
Aquí sí se utiliza una heurística admisible, diseñada específicamente para Unblock Me. La función heurística
h(n) se define como el número de bloques que obstruyen la salida del bloque rojo más un movimiento adicional del propio bloque rojo.

Ejemplo: si el bloque rojo está a 3 casillas de la salida y hay 2 bloques bloqueándolo, la heurística sería 
h(n)=2+1=3.Esta heurística nunca sobreestima el costo real (es admisible), lo que garantiza que A* encuentre siempre soluciones óptimas en cuanto a cantidad de movimientos. 
Al mismo tiempo, le da una “guía” al algoritmo, que prioriza expandir primero los estados más prometedores según f(n)=g(n)+h(n). Gracias a esto, en la práctica expande menos nodos que BFS.

# Los resultados son aleatorios, sin embargo cumple con su objetivo de comparación entre A* y BFS 
#Estados nivel fácil (6x6)
Estado inicial 1:
N° nodos revisados A*:1
N° nodos revisados BFS:1

Estado inicial 2:
N° nodos revisados A*:2
N° nodos revisados BFS:9

Estado inicial 3:
N° nodos revisados A*:7
N° nodos revisados BFS:78

Estado inicial 4:
N° nodos revisados A*:8
N° nodos revisados BFS:288

Estado inicial 5:
N° nodos revisados A*:9
N° nodos revisados BFS:583


#Estados nivel medio (8x8)
Estado inicial 1:
N° nodos revisados A*:1
N° nodos revisados BFS:1

Estado inicial 2:
N° nodos revisados A*:4
N° nodos revisados BFS:11

Estado inicial 3:
N° nodos revisados A*:5
N° nodos revisados BFS:129

Estado inicial 4:
N° nodos revisados A*:8
N° nodos revisados BFS:884


#Estados nivel difícil (20x20)
Estado inicial 1:
N° nodos revisados A*:1
N° nodos revisados BFS:1

Estado inicial 2:
N° nodos revisados A*:2
N° nodos revisados BFS:400001 (sin solución)

Estado inicial 3:
N° nodos revisados A*:23
N° nodos revisados BFS:400001 (sin solución)

Estado inicial 4:
N° nodos revisados A*:24
N° nodos revisados BFS:400001 (sin solución)

Estado inicial 5
N° nodos revisados A*:25
N° nodos revisados BFS:400001 (sin solución)
(recalcar que para llegar al resultado de nodos en BFS con el nivel difícil, las pruebas varían
de 30 minutos, 1 hora áproximadamente)

---