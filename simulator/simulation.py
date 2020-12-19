import pygame

green = (0, 255, 0)
blue = (0, 0, 128)
yellow = (255, 255, 128)
white = (255, 255, 255)
black = (0, 0, 0)


class Simulation:
    def __init__(self, config):
        pygame.init()
        print("simultaion")
        self.config = config
        self.screen = pygame.display.set_mode([750, 800])
        self.x_unit = 750 / config['sim']['width']
        self.y_unit = 750 / config['sim']['height']

    def render(self, hunters, preys, simulation_time=0, hunter_alive=0, hunter_total=0, prey_alive=0, prey_total=0,
               end=False):
        font = pygame.font.Font('freesansbold.ttf', 14)
        text_general = font.render('General:', True, yellow, black)
        text_timer = font.render('t = ' + str(simulation_time), True, white, black)
        text_field = font.render('field: ' + str(self.config['sim']['width']) + 'x' + str(self.config['sim']['height']),
                                 True, white, black)

        text_hunter = font.render('Hunters:', True, yellow, black)
        text_hunter_alive = font.render('hunter_alive: ' + str(hunter_alive), True, white, black)
        text_hunter_total = font.render('hunter_total: ' + str(hunter_total), True, white, black)
        text_hunter_start = font.render('hunter_start: ' + str(self.config["num_hunters"]), True, white, black)
        text_hunter_m_age = font.render('hunter_m_age: ' + str(self.config["hunters"]["max_age"]), True, white, black)

        text_prey = font.render('Prey:', True, yellow, black)
        text_prey_alive = font.render('prey_alive: ' + str(prey_alive), True, white, black)
        text_prey_total = font.render('prey_total: ' + str(prey_total), True, white, black)
        text_prey_start = font.render('prey_start: ' + str(self.config["num_preys"]), True, white, black)
        text_prey_m_age = font.render('prey_m_age: ' + str(self.config["preys"]["max_age"]), True, white, black)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        self.screen.fill((255, 255, 255))

        surf = pygame.Surface((750, 50))
        surf.fill(black)
        self.screen.blit(surf, (0, 750))

        textRect = text_general.get_rect()
        textRect.topleft = (10, 750)
        self.screen.blit(text_general, textRect)

        textRect = text_timer.get_rect()
        textRect.topleft = (15, 765)
        self.screen.blit(text_timer, textRect)

        textRect = text_field.get_rect()
        textRect.topleft = (15, 780)
        self.screen.blit(text_field, textRect)

        textRect = text_hunter.get_rect()
        textRect.topleft = (160, 750)
        self.screen.blit(text_hunter, textRect)

        textRect = text_hunter_alive.get_rect()
        textRect.topleft = (165, 765)
        self.screen.blit(text_hunter_alive, textRect)

        textRect = text_hunter_total.get_rect()
        textRect.topleft = (165, 780)
        self.screen.blit(text_hunter_total, textRect)

        textRect = text_hunter_start.get_rect()
        textRect.topleft = (285, 764)
        self.screen.blit(text_hunter_start, textRect)

        textRect = text_hunter_m_age.get_rect()
        textRect.topleft = (285, 780)
        self.screen.blit(text_hunter_m_age, textRect)

        textRect = text_prey.get_rect()
        textRect.topleft = (450, 750)
        self.screen.blit(text_prey, textRect)

        textRect = text_prey_alive.get_rect()
        textRect.topleft = (455, 765)
        self.screen.blit(text_prey_alive, textRect)

        textRect = text_hunter_total.get_rect()
        textRect.topleft = (455, 780)
        self.screen.blit(text_prey_total, textRect)

        textRect = text_hunter_start.get_rect()
        textRect.topleft = (575, 764)
        self.screen.blit(text_prey_start, textRect)

        textRect = text_hunter_m_age.get_rect()
        textRect.topleft = (575, 780)
        self.screen.blit(text_prey_m_age, textRect)

        if not len(hunters) == 0:
            for observation in hunters:
                x, y = observation
                # print(x.item(), y, observation)
                surf = pygame.Surface((self.x_unit, self.y_unit))
                surf.fill((255, 127, 80))
                self.screen.blit(surf, (x.item() * self.x_unit, y.item() * self.y_unit))
        if not len(preys) == 0:
            for observation in preys:
                x, y = observation
                surf = pygame.Surface((self.x_unit, self.y_unit))
                surf.fill((220, 220, 220))
                self.screen.blit(surf, (x * self.x_unit, y * self.y_unit))
        pygame.display.flip()
        while end:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    end = False
        return True

    def quit(self):
        pygame.quit()
        # self.monitor.plot()
