import pygame


def main():
    screen_width = 1200
    screen_height = 800
    clock = pygame.time.Clock()
    pygame.init()
    font = pygame.font.Font(pygame.font.get_default_font(), 20)

    screen = pygame.display.set_mode((screen_width, screen_height))

    running = True

    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    BLUE = (0, 0, 255)
    screen.fill(BLACK)


    mode_de_jeu = False
    parametre_du_mode_de_jeu = 0
    dico = {
        True: ('contre la montre', [(0.5, '30s'), (1, '1min'), (2, '2min')]),
        False: ("nombre d'equation", [(10, '10'), (20, '20'), (50, '50')])
            }

    t = dico[mode_de_jeu]
    text1 = font.render(t[0], True, WHITE)
    text1_rect = text1.get_rect()
    text1_rect.center = (600,512)
    screen.blit(text1, text1_rect)

    text2 = font.render(t[1][parametre_du_mode_de_jeu][1], True, WHITE)
    text2_rect = text1.get_rect()
    text2_rect.center = (680,600)
    screen.blit(text2, text2_rect)



    img_play = pygame.image.load("images/bouton_erase_off.png")
    img_play_on = pygame.image.load("images/bouton_erase_on.png")
    play_bouton = img_play.get_rect()
    play_bouton.left, play_bouton.top = screen_width//2 - \
        128, screen_height//2 - 77//2

    img_left_arrow = pygame.image.load("images/left_arrow.png")
    img_left_arrow_on = pygame.image.load("images/left_arrow_on.png")
    leftarrow1, leftarrow2 = img_left_arrow.get_rect(), img_left_arrow.get_rect()
    leftarrow1.left, leftarrow1.top = 450 - 52//2, 510 - 46//2
    leftarrow2.left, leftarrow2.top = 450 - 52//2, 600 - 46//2

    img_right_arrow = pygame.image.load("images/right_arrow.png")
    img_right_arrow_on = pygame.image.load("images/right_arrow_on.png")
    rightarrow1, rightarrow2 = img_right_arrow.get_rect(), img_right_arrow.get_rect()
    rightarrow1.left, rightarrow1.top = 750 - 52//2, 510 - 46//2
    rightarrow2.left, rightarrow2.top = 750 - 52//2, 600 - 46//2

    while running:

        screen.fill(BLACK)
        screen.blit(text1, text1_rect)
        screen.blit(text2, text2_rect)

        pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    if leftarrow1.collidepoint(pos) or rightarrow1.collidepoint(pos):
                        mode_de_jeu = bool(1-mode_de_jeu)
                        text1 = font.render(dico[mode_de_jeu][0], True, WHITE)
                    if leftarrow2.collidepoint(pos):
                        parametre_du_mode_de_jeu = (parametre_du_mode_de_jeu-1) % 3
                        text2 = font.render(dico[mode_de_jeu][1][parametre_du_mode_de_jeu][1], True, WHITE)
                    if rightarrow2.collidepoint(pos):
                        parametre_du_mode_de_jeu = (parametre_du_mode_de_jeu+1) % 3
                        text2 = font.render(dico[mode_de_jeu][1][parametre_du_mode_de_jeu][1], True, WHITE)

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

        pygame.display.flip()


main()