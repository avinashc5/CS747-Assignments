import time
import os 
import argparse
import pygame

### Visualize gamplay 
class MiniChessGUI:
    def __init__(self, FENs, res, tframe):
        self.fens=  FENs
        self.res = res
        self.tframe = tframe
        temp = FENs[0].split(" ")[0].split("/")
        ranks = len(temp)
        files = 0
        for ch in temp[0]:
            if ch.isalpha(): files+=1
            elif ch.isdigit(): files+=int(ch)
            else: raise ValueError('FEN parse error')
        self.dims = [files,ranks]
        
        # PyGame setup
        pygame.init()
        self.square_size = 120
        self.width = self.dims[0] * self.square_size   # 4 cols wide
        self.height = self.dims[1] * self.square_size + 80  # 5 rows + input bar
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("5×4 MiniChess (Keyboard Play)")
        self.font = pygame.font.SysFont("Arial", 32)
        self.text_input = ""

        # Load piece images
        self.pieces = {}
        folder = "assets/pieces"
        for name in ["P", "R", "N", "B", "Q", "K"]:
            self.pieces["w" + name] = pygame.transform.scale(
                pygame.image.load(os.path.join(folder, f"w{name}.png")),
                (self.square_size, self.square_size)
            )
            self.pieces["b" + name] = pygame.transform.scale(
                pygame.image.load(os.path.join(folder, f"b{name}.png")),
                (self.square_size, self.square_size)
            )

    # ---------- FEN → matrix ----------
    def fen_to_matrix(self, fen):
        rows = fen.split(" ")[0].split("/")
        board = []
        for row in rows:
            expanded = ""
            for ch in row:
                expanded += "." * int(ch) if ch.isdigit() else ch
            board.append(list(expanded))
        return board  

    # ---------- Drawing ----------
    def draw_board(self, fen):
        board = self.fen_to_matrix(fen)
        board = board[::-1]
        # Draw squares (rank 5 at top → rank 1 at bottom)
        for y in range(self.dims[1]):
            for x in range(self.dims[0]):
                color = (240, 217, 181) if (x + y) % 2 == 0 else (181, 136, 99)
                rect = pygame.Rect(
                    x * self.square_size,
                    (self.dims[1] - 1 - y) * self.square_size,
                    self.square_size,
                    self.square_size
                )
                pygame.draw.rect(self.screen, color, rect)

                piece = board[y][x]
                if piece != ".":
                    img_key = ("w" if piece.isupper() else "b") + piece.upper()
                    self.screen.blit(self.pieces[img_key], rect.topleft)

        # Input bar
        pygame.draw.rect(self.screen, (50, 50, 50), (0, self.height - 80, self.width, 80))
        
        pygame.display.flip()
        pygame.display.flip()

    def _run(self):
        clock = pygame.time.Clock()

        index = 0                      # Current frame index
        paused = False                 # Whether playback is paused
        num_frames = len(self.fens)    # Total number of FEN positions
        res_printed = False
        running = True

        while running:
            for event in pygame.event.get():
                # Quit window
                if event.type == pygame.QUIT:
                    running = False

                # Keyboard controls
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:      # Pause/resume toggle
                        paused = not paused

                    elif event.key == pygame.K_RIGHT:    # Next frame
                        if paused:
                            index = min(index + 1, num_frames - 1)

                    elif event.key == pygame.K_LEFT:     # Previous frame
                        if paused:
                            index = max(index - 1, 0)

            # --- Draw current frame ---
            self.draw_board(self.fens[index])

            # If paused, show "Paused" indicator below board
            if paused:
                text = self.font.render("Paused", True, (255, 255, 255))
                self.screen.blit(text, (20, self.height - 60))

            # Display result text after last frame
            if index == num_frames - 1:
                gameend = {'-1': 'Black', '0': 'Draw', '1': 'White'}
                if self.res is not None:
                    text = self.font.render(gameend[self.res], True, (255, 255, 255))
                else:
                    text = self.font.render("No result", True, (255, 255, 255))
                self.screen.blit(text, (20, self.height - 60))
                if not res_printed:
                    print("THE END OF GAME IS ---", str(self.res).upper())
                    res_printed = True

            # Update display
            pygame.display.flip()

            # --- Automatic advance if not paused ---
            if not paused:
                time.sleep(self.tframe)
                if index < num_frames - 1:
                    index += 1

            clock.tick(30)

        pygame.quit()



# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fens_path", type=str, help="path to file containing FEN of the game history in each line, with final line being result of game (white=1/black=-1/draw=0)")
    parser.add_argument("--tframe", type=float, default=1.3, help="time between frames")
    args = parser.parse_args()

    fen_path = args.fens_path
    with open(fen_path, 'r') as f:
        fens = list(f.readlines())
    res=None
    start_player = None
    if fens[0].strip() in ['0','1']:
        start_player = fens[0].strip() 
        fens = fens[1:]
    while fens[-1].strip() == "":
        fens = fens[:-1]
    res = ""
    if fens[-1].strip() in ['0', '1', '-1']:
        res = fens[-1].strip()
        fens = fens[:-1]
    player_no = {'0':'MnMx','1':'NN'}
    game_res_no = {'0':'Draw', '1':'White', '-1':'Black', '':'No result'}
    if start_player:
        print('Start player: ',player_no[start_player])
    print('Game result: ', game_res_no[res])

    frame_time = args.tframe
    game = MiniChessGUI(fens,res,frame_time)
    game._run()



if __name__ == "__main__":
    main()
