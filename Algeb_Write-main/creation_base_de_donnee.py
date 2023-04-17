import pygame
import pygame.display
from pygame.locals import (
    K_ESCAPE,
    KEYDOWN,
    K_SPACE,
    K_b,
    K_RETURN

)

import numpy as np
from random import randint
import pickle
from copy import copy
import pickle


def remise_a_bonne_taille(array):
    '''transforme une array(qui forme une image) sous une array plus petite (28x28 car le reseau de neurone a été entrainé a cette taille) (qui forme toujours l'image)'''
    x, y = array.shape
    mult_x, mult_y = x // 28, y // 28
    xf, yf = 28, 28

    array1 = np.zeros(shape=(x, yf))
    array2 = np.zeros(shape=(xf, yf))
    for k in range(yf):
        for i in range(x):
            array1[i][k] = np.sum(
                np.array(array[i][k * mult_y: (k * mult_y) + mult_y])) // mult_y
    for k in range(yf):
        for i in range(xf):
            array2[i][k] = 255 - np.sum(np.array(array1.T[k]
                                        [i * mult_x: (i * mult_x) + mult_x])) // mult_x
    return array2


def mise_a_multiple(array):
    '''prend un array et y ajoute des marge pour que la hauteur et la largeur de l'array soit multiple de 28 (et carré)'''
    n, m = array.shape
    taille_carre = max(n, m)

    taille_corecte = ((taille_carre//28 + (taille_carre % 28 != 0)) + 2) * 28
    c, d = taille_corecte-n, taille_corecte-m
    marge_haut = c // 2 + (c % 2 == 1)
    haut = np.full((marge_haut, m), 255)
    array = np.vstack((haut, array))
    marge_bas = c // 2
    bas = np.full((marge_bas, m), 255)
    array = np.vstack((array, bas))
    marge_droit = d // 2 + (d % 2 == 1)
    droit = np.full((taille_corecte, marge_droit), 255)
    array = np.hstack((droit, array))
    marge_gauche = d // 2
    gauche = np.full((taille_corecte, marge_gauche), 255)
    array = np.hstack((array, gauche))
    return array


def recentrer(array):
    '''array de taille 252 par 252 -> enleve tous les rangé formé que de 255'''
    for i, ligne in enumerate(array):
        n = len(ligne)
        if False in (np.array([255 for k in range(n)]) == ligne):
            array = array[i:]
            break
    for i, ligne in enumerate(array[::-1]):
        i = len(array) - i
        n = len(ligne)
        if False in (np.array([255 for k in range(n)]) == ligne):
            array = array[:i]
            break
    array = array.T
    for i, colone in enumerate(array):
        n = len(colone)
        if False in (np.array([255 for k in range(n)]) == colone):
            array = array[i:]
            break
    for i, colone in enumerate(array[::-1]):
        i = len(array) - i
        n = len(colone)
        if False in (np.array([255 for k in range(n)]) == colone):
            array = array[:i]
            break
    array = array.T
    return array


def mise_en_forme(array):
    '''prend une array de taille 252x252 qui forme une image
    et la transforme en array de 28x28 qui forme toujours la meme image'''
    a = recentrer(array)
    b = mise_a_multiple(a)
    c = remise_a_bonne_taille(b)
    return c


def main():
    screen_width = 1200
    screen_height = 800
    clock = pygame.time.Clock()
    pygame.init()
    fond = pygame.image.load('images/fond2.png')
    zones = pygame.image.load('images\zones.png')

    screen = pygame.display.set_mode((screen_width, screen_height))
    zone = pygame.Surface((252, 252))
    zone2 = pygame.Surface((252, 252))
    pygame.font.init()
    font_path = "font\Typo_Round_Bold_Demo.otf"
    font150 = pygame.font.Font(font_path, 150)
    fonct50 = pygame.font.Font(font_path, 50)

    running = True

    nb = (randint(0, 9), randint(0, 9))

    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    screen.blit(fond, (0, 0))
    text = font150.render(str(nb[0]) + '  ' + str(nb[1]), True, BLACK)
    screen.fill(WHITE)
    screen.blit(text, (180, 260))
    zone.fill(WHITE)
    zone2.fill(WHITE)

    # angle haut gauche des trou poour les zone
    trou = ((634, 274), (914, 274))
    x, y = trou[0]
    screen.blit(zone, trou[0])
    screen.blit(zone2, trou[1])
    zone.get_rect(topleft=trou[0])
    screen.blit(fond, (0, 0))
    screen.blit(text, (180, 270))

    im = pickle.load(open('base_de_donnee/test-image.pkl', 'rb'))
    lm = pickle.load(open('base_de_donnee/test-labels.pkl', 'rb'))
    nb_image = im.shape[1]
    im = im.T[:1].T
    lm = lm.T[:1].T
    while running:
        clock.tick(360)

        for event in pygame.event.get():
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False

                if event.key == K_RETURN:
                    bleu = copy(pygame.surfarray.pixels_blue(zone))
                    bleu2 = copy(pygame.surfarray.pixels_blue(zone2))
                    image = mise_en_forme(bleu.T)
                    if 0 in bleu:
                        im = np.hstack((im, image.reshape((784, 1))))
                        li = np.zeros((10, 1))
                        li[nb[0]][0] = 1
                        lm = np.hstack((lm, li))
                    image2 = mise_en_forme(bleu2.T)
                    if 0 in bleu2:
                        im = np.hstack((im, image2.reshape((784, 1))))
                        li = np.zeros((10, 1))
                        li[nb[0]][0] = 1
                        lm = np.hstack((lm, li))
                    zone.fill(WHITE)
                    zone2.fill(WHITE)
                    nb = (randint(0, 9), randint(0, 9))
                    text = font150.render(
                        str(nb[0]) + '  ' + str(nb[1]), True, BLACK)
                    nb_image += 2
                    screen.blit(fond, (0, 0))
                    text2 = fonct50.render(str(nb_image), True, BLACK)
                    screen.blit(text2, (870, 183))
                    screen.blit(text, (180, 270))

            elif event.type == pygame.MOUSEMOTION:
                if event.buttons[0]:  # Left mouse button down.
                    if trou[1][0] < event.pos[0]:  # trou - x = diference entre les deux trous
                        pygame.draw.circle(
                            zone2, BLACK, (event.pos[0] - trou[1][0], event.pos[1] - y), 10)
                    else:
                        pygame.draw.circle(
                            zone, BLACK, (event.pos[0] - x, event.pos[1] - y), 10)
                if event.buttons[2]:
                    if zone.get_rect(topleft=trou[0]).collidepoint(pygame.mouse.get_pos()):
                        zone.fill(WHITE)
                    if zone2.get_rect(topleft=trou[1]).collidepoint(pygame.mouse.get_pos()):
                        zone2.fill(WHITE)

        screen.blit(zone, trou[0])
        screen.blit(zone2, trou[1])
        screen.blit(zones, (630, 270))

        pygame.display.flip()

    return im, lm


im, lm = main()

im_o = pickle.load(open('base_de_donnee/test-image.pkl', 'rb'))
lm_o = pickle.load(open('base_de_donnee/test-labels.pkl', 'rb'))
f = open("base_de_donnee/test-image.pkl", "wb")
pickle.dump(np.hstack((im_o, im)), f)
f.close()
f = open("base_de_donnee/test-labels.pkl", "wb")
pickle.dump(np.hstack((lm_o, lm)), f)
f.close()
