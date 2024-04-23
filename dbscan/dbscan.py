import pygame
import numpy as np

def query_region(D, P, eps):
    neighbors = []
    for Pn in range(0, len(D)):
        if np.linalg.norm(D[P] - D[Pn]) < eps:
           neighbors.append(Pn)
    return neighbors

def expand_cluster(D, labels, P, NeighborPts, C, eps, MinPts):
    labels[P] = C
    i = 0
    while i < len(NeighborPts):
        Pn = NeighborPts[i]
        if labels[Pn] == -1:
           labels[Pn] = C
        elif labels[Pn] == 0:
            labels[Pn] = C
            PnNeighborPts = query_region(D, Pn, eps)
            if len(PnNeighborPts) >= MinPts:
                NeighborPts = NeighborPts + PnNeighborPts
        i += 1

def dbscan(D, eps, MinPts):
    labels = [0]*len(D)
    C = 0
    for P in range(0, len(D)):
        if not (labels[P] == 0):
           continue
        NeighborPts = query_region(D, P, eps)
        if len(NeighborPts) < MinPts:
            labels[P] = -1
        else:
           C += 1
           expand_cluster(D, labels, P, NeighborPts, C, eps, MinPts)
    return labels

pygame.init()
screen = pygame.display.set_mode((800, 600))
points = []
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            points.append(pygame.mouse.get_pos())
    screen.fill((0, 0, 0))
    for point in points:
        pygame.draw.circle(screen, (255, 255, 255), point, 5)
    pygame.display.flip()
pygame.quit()

labels = dbscan(np.array(points), eps=100, MinPts=5)
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)]
pygame.init()
screen = pygame.display.set_mode((800, 600))
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    screen.fill((0, 0, 0))
    for i, point in enumerate(points):
        pygame.draw.circle(screen, colors[labels[i]], point, 5)
    pygame.display.flip()
pygame.quit()
