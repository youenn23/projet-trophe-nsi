##importation bibliotèques
import pygame
import pygame.display
from pygame.locals import (
    K_ESCAPE,
    KEYDOWN,
    K_SPACE,
    K_BACKSPACE,
)
import numpy as np
from random import randint
import pickle
from copy import copy

##classe permettant d'utilisé le reseau de neurone pre-entrainé
class utilisation_reseau_de_neurone:
    def __init__(self):
        self.parametres = pickle.load(open('parametres/file3.pkl', 'rb'))      
        self.nombre_de_couche = len(self.parametres) // 2

    def predict(self, X):
        activations = self.forward_propagation(X)
        Af = activations['A' + str(self.nombre_de_couche)]    
        return Af

    def forward_propagation(self, X):
        activations = {'A0': X}
        for c in range(1, self.nombre_de_couche + 1):
            Z = self.parametres['W' + str(c)].dot(activations['A' + str(c - 1)]) + self.parametres['b' + str(c)]
            activations['A' + str(c)] = 1/ ( 1+ np.exp(-Z))
        return activations


##fonctions creation calculs
def creation_calcul() -> str:
    '''crée un calcul avec deux nombre entre 0 et 9 avec 1 operateur logique
    le calcul doit etre valide, et ne crée pas de nombre négatif'''
    operateur  = ['×','/','-','+']
    a =  randint(0,9)
    b = randint(1,9)
    c = operateur[randint(0,3)]
    if c == '/':
        while a%b != 0 :
            b = randint(1,9)
    if c == '-' and a<b:
        a,b = b,a
    calcul = str(a)  + c +  str(b)
    return calcul

def resultat(chaine:str) -> int: 
    '''renvoie le resultat (en int) d'un calcul simple (sous forme d'une chaine)'''
    a = int(chaine[0])
    b = int(chaine[2])
    c = chaine[1]
    if c == '×':
        return a*b
    if c == '/':
        return int(a/b)
    if c == '+':
        return a+b
    if c == '-':
        return a-b   

##fonctons pour lire l'image du chiffre
def recentrer(array):
    '''array de taille 252 par 252 -> enleve tous les rangé formé que de 255'''
    for i, ligne in enumerate(array):
        n = len(ligne)
        if False in ( np.array([255 for k in range(n)]) == ligne):
            array= array[i:]
            break
    for i, ligne in enumerate(array[::-1]):
        i = len(array) - i
        n = len(ligne)
        if False in ( np.array([255 for k in range(n)]) == ligne):
            array= array[:i]
            break
    array= array.T
    for i, colone in enumerate(array):
        n = len(colone)
        if False in ( np.array([255 for k in range(n)]) == colone):
            array= array[i:]
            break
    for i, colone in enumerate(array[::-1]):
        i = len(array) - i
        n = len(colone)
        if False in ( np.array([255 for k in range(n)]) == colone):
            array= array[:i]
            break
    array= array.T
    return array

def mise_a_multiple(array):
    '''prend un array et y ajoute des marge pour que la hauteur et la largeur de l'array soit multiple de 28 (et carré)'''
    n,m = array.shape
    taille_carre = max(n,m)

    taille_corecte = ((taille_carre//28 + (taille_carre%28 != 0))  + 2) * 28
    c, d = taille_corecte-n , taille_corecte-m
    marge_haut = c // 2 + ( c%2 == 1)  
    haut = np.full((marge_haut,m), 255)
    array = np.vstack((haut, array))
    marge_bas = c // 2  
    bas = np.full((marge_bas,m), 255)
    array = np.vstack((array,bas))   
    marge_droit = d // 2 + ( d%2 == 1)
    droit = np.full((taille_corecte,marge_droit), 255)
    array = np.hstack((droit, array))
    marge_gauche = d // 2  
    gauche = np.full((taille_corecte,marge_gauche), 255)
    array = np.hstack((array, gauche))   
    return array

def remise_a_bonne_taille(array):
    '''transforme une array(qui forme une image) sous une array plus petite
    (28x28 car le reseau de neurone a été entrainé a cette taille) (qui forme toujours l'image)'''
    x, y = array.shape
    mult_x , mult_y =  x //28 , y // 28
    xf , yf = 28, 28

    array1 = np.zeros(shape=(x,yf))
    array2 = np.zeros(shape=(xf,yf))
    for k in range(yf):
        for i in range(x):
            array1[i][k] = np.sum(np.array(array[i][ k * mult_y : (k * mult_y) + mult_y])) //mult_y
    for k in range(yf):
        for i in range(xf):
            array2[i][k] = 255 - np.sum(np.array(array1.T[k][ i * mult_x : (i * mult_x) + mult_x])) //mult_x
    return array2

def mise_en_forme(array):
    '''prend une array de taille 252x252 qui forme une image
    et la transforme en array de 28x28 qui forme toujours la meme image'''
    a = recentrer(array)
    b = mise_a_multiple(a)
    c = remise_a_bonne_taille(b)
    return c

def lisible(image,reseau) -> str: 
    '''renvoie le nombre lisible'''
    return str(np.argmax(reseau.predict(image)))



def main():
    reseau = utilisation_reseau_de_neurone()

    clock = pygame.time.Clock()
    pygame.init()

    #importations d'images
    logo = pygame.image.load("images\logo32x32.png")
    fond = pygame.image.load('images/fond2.png')
    zones = pygame.image.load('images\zones.png')
    zone_resultat = pygame.image.load('images/zone_resultat.png')
    valide = pygame.image.load('images/bouton_valide.png')
    invalide = pygame.image.load('images/bouton_invalide.png')
    img_erase_bouton = pygame.image.load("images/bouton_erase_off1.png")
    img_erase_bouton_on = pygame.image.load("images/bouton_erase_on1.png")
    img_submit_bouton = pygame.image.load("images/bouton_submit_off.png")
    img_submit_bouton_on = pygame.image.load("images/bouton_submit_on.png")
    zone_score = pygame.image.load('images/zone_score.png')
    zone_timer = pygame.image.load('images/zone_timer.png')
    img_left_arrow = pygame.image.load("images/left_arrow.png")
    img_left_arrow_on = pygame.image.load("images/left_arrow_on.png")
    img_right_arrow = pygame.image.load("images/right_arrow.png")
    img_right_arrow_on = pygame.image.load("images/right_arrow_on.png")


    #taille de du jeu et icone+nom du jeu
    screen_width = 1200
    screen_height = 800
    pygame.display.set_icon(logo)
    pygame.display.set_caption("projet")
    screen = pygame.display.set_mode((screen_width,screen_height))

    ##mise en place des zones de dessin
    coo_zones = (630, 270)
    zone = pygame.Surface((252, 252))
    zone2 = pygame.Surface((252, 252))

    #police d'ecriture
    pygame.font.init()
    font_path = "font\Typo Grotesk Rounded Demo.otf"
    font150 = pygame.font.Font(font_path, 150)
    font100 = pygame.font.Font(font_path, 100)
    font45 = pygame.font.Font(font_path, 48)
    font30 = pygame.font.Font(font_path, 30 )

    ##couleurs
    Beige = (255,224,211)
    vert_foncé = (0,60,45)
    
    ##bouls
    running, menu, jeu, fini = True, True, False, False

    ##selection mode de jeu
    mode_de_jeu = False #True = contre la montre, False = serie
    parametre_du_mode_de_jeu = 10 #contre la montre : (0.5, 1, 2) ; serie : (10, 20, 50)
    index_parametre_du_mode_de_jeu = 0
    dico = {
        True: ( ' contre la montre', [(0.5, '30 seconde'), (1, ' 1 minutes'), (2, ' 2 minutes')]),
        False: ("nombre d'equation", [(10, ' 10 équations'), (20, '20 équations'), (50, '50 équations')])
            }
    #texte lors du choix du mode de jeu
    t = dico[mode_de_jeu]
    text1 = font30.render(t[0], True, Beige)
    text1_rect = text1.get_rect()
    text1_rect.center = (600,512)
    text2 = font30.render(t[1][index_parametre_du_mode_de_jeu][1], True, Beige)
    text2_rect = text1.get_rect()
    text2_rect.center = (650,600)
    

    ##mise en place des calculs
    calcul =  creation_calcul()
    calcul_secondaire = creation_calcul()
    screen.blit(fond, (0,0))
    text_calcul = font150.render(calcul, True, vert_foncé)
    text_calcul_secondaire = font100.render(calcul_secondaire, True, vert_foncé)

    ##variable score + recupération des records
    score = 0
    record = pickle.load(open('parametres/record.pkl', 'rb')) 

    text_score = font30.render("score: " + str(score), True, Beige)
    t_score = text_score.get_rect()
    t_score.center = (1100,20)

    text_hscore = font30.render("record: " + str(record[mode_de_jeu][parametre_du_mode_de_jeu]), True, Beige)
    t_hscore = text_hscore.get_rect()
    t_hscore.center = (1090,50)
    

    pygame.time.set_timer(pygame.USEREVENT, 1000)


    ##mise en place boutons du menu
    img_play = pygame.image.load("images/bouton_erase_off.png")
    img_play_on = pygame.image.load("images/bouton_erase_on.png")
    play_bouton = img_play.get_rect()
    play_bouton.left, play_bouton.top = screen_width//2 - 252//2, screen_height//2 - 77//2

    leftarrow1, leftarrow2 = img_left_arrow.get_rect(), img_left_arrow.get_rect()
    leftarrow1.left, leftarrow1.top = 400 - 52//2, 510 - 46//2
    leftarrow2.left, leftarrow2.top = 400 - 52//2, 600 - 46//2

    rightarrow1, rightarrow2 = img_right_arrow.get_rect(), img_right_arrow.get_rect()
    rightarrow1.right, rightarrow1.top = 800 + 52//2, 510 - 46//2
    rightarrow2.right, rightarrow2.top = 800 + 52//2, 600 - 46//2



    ##affichage des zones de dessin
    trou = ((760-126,400-126),(1040-126,400-126)) #angle haut gauche des trou poour les zone
    x,y = trou[0]

    
    ##affichage des calculs
    c0 = text_calcul.get_rect()
    c0.center = (300,320)
    
    c1 = text_calcul_secondaire.get_rect()
    c1.center = (300,480)

    ##mise en place des bouton pour effacer et pour valider le résultat
    erase_bouton1, erase_bouton2 , submit_bouton = img_erase_bouton.get_rect(), img_erase_bouton.get_rect() , img_submit_bouton.get_rect()
    erase_bouton1.left, erase_bouton2.left , submit_bouton.left  = 634, 914, 634
    erase_bouton1.top, erase_bouton2.top , submit_bouton.top = 545, 545, 650

    ##variable de comparable pour les calculs
    result = None
    lisibl = None

    ##affiche ce qui est lisible
    text3 = font45.render('', True, Beige)

    NOM = font150.render("Algeb'write", True, Beige)

    while running:
        clock.tick(180) ##nombre de fps

        if menu:
            screen.fill(vert_foncé)
            screen.blit(text1, text1_rect)
            screen.blit(NOM, (220,100))
            screen.blit(text2, text2_rect)
            pos = pygame.mouse.get_pos()
            ##effet permettant au joueur de voir lorsqu'il passe sur un bouton
            if play_bouton.collidepoint(pos):
                screen.blit(img_play_on,  play_bouton)
            else:
                screen.blit(img_play,  play_bouton)
            if leftarrow1.collidepoint(pos):
                screen.blit(img_left_arrow_on,  leftarrow1)
            else:
                screen.blit(img_left_arrow,  leftarrow1)
            if leftarrow2.collidepoint(pos):
                screen.blit(img_left_arrow_on,  leftarrow2)
            else:
                screen.blit(img_left_arrow,  leftarrow2)
            if rightarrow1.collidepoint(pos):
                screen.blit(img_right_arrow_on,  rightarrow1)
            else:
                screen.blit(img_right_arrow,  rightarrow1)
            if rightarrow2.collidepoint(pos):
                screen.blit(img_right_arrow_on,  rightarrow2)
            else:
                screen.blit(img_right_arrow,  rightarrow2)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        running = False 

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        if play_bouton.collidepoint(pos):
                            #passage du menu au jeu
                            menu = False
                            jeu = True
                            screen.fill(Beige)
                            screen.blit(fond, (0,0))
                            zone.fill(Beige)
                            zone2.fill(Beige)
                            screen.blit(text_calcul_secondaire, c1)
                            screen.blit(text_calcul, c0)
                            screen.blit(text_score, t_score)
                            screen.blit(zone, trou[0])
                            screen.blit(zone2, trou[1])
                            parametre_du_mode_de_jeu = dico[mode_de_jeu][1][index_parametre_du_mode_de_jeu][0]
                            a = record[mode_de_jeu][parametre_du_mode_de_jeu]
                            if a == float('inf'):
                                text_hscore = font30.render("record: Ø" , True, Beige)
                            else:
                                text_hscore = font30.render("record: " + str(a) , True, Beige)
                            screen.blit(text_hscore, t_hscore)
                            ##temps:
                            if mode_de_jeu:
                                print(str(int(parametre_du_mode_de_jeu * 60)))
                                decompte, text_decompte= int(parametre_du_mode_de_jeu * 60) , str(int(parametre_du_mode_de_jeu * 60))
                            else:
                                decompte, text_decompte= 0,'0'                    
                                nb_dequation = 0
                        
                        ##choix du mode de jeu
                        if leftarrow1.collidepoint(pos) or rightarrow1.collidepoint(pos):
                            mode_de_jeu = bool(1-mode_de_jeu)
                            text1 = font30.render(dico[mode_de_jeu][0], True, Beige)
                            text2 = font30.render(dico[mode_de_jeu][1][index_parametre_du_mode_de_jeu][1], True, Beige)
                        if leftarrow2.collidepoint(pos):
                            index_parametre_du_mode_de_jeu = (index_parametre_du_mode_de_jeu-1) % 3
                            text2 = font30.render(dico[mode_de_jeu][1][index_parametre_du_mode_de_jeu][1], True, Beige)
                        if rightarrow2.collidepoint(pos):
                            index_parametre_du_mode_de_jeu = (index_parametre_du_mode_de_jeu+1) % 3
                            text2 = font30.render(dico[mode_de_jeu][1][index_parametre_du_mode_de_jeu][1], True, Beige)


            pygame.display.flip()


        elif jeu == True:       

            ##permet de gerer l'arret du jeu
            pos = pygame.mouse.get_pos()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                ##timer
                if event.type == pygame.USEREVENT:
                    if mode_de_jeu: 
                        decompte -= 1
                        text_decompte = str(decompte)
                        if decompte == 0:
                            running = False
                            jeu = False
                            fini =True
                    else:
                        score += 1
                        text_score = font30.render("score: " + str(score), True, Beige)
                        text_decompte = str('')

                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        running = False  

                    ## si espace appuyer -> changement de calculs
                    if event.key == K_SPACE:
                        zone.fill(Beige)
                        zone2.fill(Beige)
                        calcul,calcul_secondaire = calcul_secondaire,creation_calcul()
                        text_calcul = font150.render(calcul, True, vert_foncé)
                        text_calcul_secondaire = font100.render(calcul_secondaire, True, vert_foncé)
                        screen.blit(fond, (0,0))
                        screen.blit(text_calcul, c0)
                        screen.blit(text_calcul_secondaire, c1)
                        if not mode_de_jeu:
                            nb_dequation += 1
                            if nb_dequation == parametre_du_mode_de_jeu:
                                jeu = False
                                fini = True


                    ##si effacer appuyer -> effacages des deux zones
                    if event.key == K_BACKSPACE   :
                        zone.fill(Beige)
                        zone2.fill(Beige)

                ##lorsque clic gauche de la souris
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        #efface une zone si le clic est sur le bouton effacer correspondant
                        if erase_bouton1.collidepoint(pos):
                            zone.fill(Beige)
                        if erase_bouton2.collidepoint(pos):
                            zone2.fill(Beige)
                        #valide ou invalide le resultat si le clic est sur le bouton valider
                        if submit_bouton.collidepoint(pos):
                            if result==  lisibl:
                                screen.blit(valide , (970, 190))
                                score += 1
                                screen.blit(zone_score, (1000,0))
                                text_score = font30.render("score: " + str(score), True, Beige)
                            else:
                                screen.blit(invalide,(970, 190) )
                        #sinon -> on lis les zones
                        else:
                            rouge = copy(pygame.surfarray.pixels_red(zone))
                            rouge2 = copy(pygame.surfarray.pixels_red(zone2))
                            a = 0 in rouge
                            b = 0 in rouge2
                            image =  mise_en_forme(rouge.T)
                            result = str(resultat(calcul))
                            if a:
                                lisibl = lisible(image.reshape((784,1)) , reseau)
                            else:
                                lisibl = ''
                            image2 = mise_en_forme(rouge2.T)
                            if b:
                                lisibl += lisible(image2.reshape((784,1)) , reseau)
                                if len(lisibl) == 1:
                                    text3 = font45.render(' ' + lisibl, True,  Beige)
                                else:
                                    text3 = font45.render(lisibl, True,  Beige)
                            else:
                                lisibl += ''
                                text3 = font45.render(lisibl, True,  Beige)
                            screen.blit(zone_resultat, (770,180))
                            screen.blit(text3, (870, 183))


                ##si clic enfoncer -> dessine dans la zone selectionner
                elif event.type == pygame.MOUSEMOTION:
                    if event.buttons[0]:  # Left mouse button down.
                        if trou[1][0] < event.pos[0]:  # trou - x = diference entre les deux trous
                            pygame.draw.circle(
                                zone2, vert_foncé, (event.pos[0] - trou[1][0], event.pos[1] - y), 10)
                        else:
                            pygame.draw.circle(
                                zone, vert_foncé, (event.pos[0] - x, event.pos[1] - y), 10)

            ##actualises les zones (donc affiche le dessins)
            screen.blit(zone, trou[0])
            screen.blit(zone2, trou[1])
            screen.blit(zones,coo_zones)
            screen.blit(zone_score, (1000,0))
            screen.blit(text_score, t_score)
            screen.blit(text_hscore, t_hscore)
            screen.blit(zone_timer, (100,100))
            screen.blit(font45.render(text_decompte, True, vert_foncé), (100,100))

            ##effet permettant au joueur de voir lorsqu'il passe sur un bouton
            if erase_bouton1.collidepoint(pos):
                screen.blit(img_erase_bouton_on, erase_bouton1)
            else:
                screen.blit(img_erase_bouton, erase_bouton1)

            if erase_bouton2.collidepoint(pos):
                screen.blit(img_erase_bouton_on, erase_bouton2)
            else:
                screen.blit(img_erase_bouton, erase_bouton2)
            
            if submit_bouton.collidepoint(pos):
                screen.blit(img_submit_bouton_on, submit_bouton)
            else: 
                screen.blit(img_submit_bouton, submit_bouton)


            pygame.display.flip()
        if fini:    
            if mode_de_jeu:
                record[mode_de_jeu][parametre_du_mode_de_jeu] = max(record[mode_de_jeu][parametre_du_mode_de_jeu],score)
            if not mode_de_jeu:
                record[mode_de_jeu][parametre_du_mode_de_jeu] = min(record[mode_de_jeu][parametre_du_mode_de_jeu],score)
            running = False
            
    return record

record = main()
f = open('parametres/record.pkl',"wb")
pickle.dump(record,f)
f.close()