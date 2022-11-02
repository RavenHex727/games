class TestPlayer:
    def __init__(self):
        self.symbol = None
        self.number = None
    
    def set_player_symbol(self, n):
        self.symbol = n
    
    def set_player_number(self, n):
        self.number = n
    
    def get_possible_moves(self, board):
        possible_moves = [(i,j) for i in range(3) for j in range(3) if board[i][j] == None]
        return possible_moves
    
    def choose_move(self, game_board):
        choices = [(i,j) for i in range(len(game_board)) for j in range(len(game_board)) if game_board[i][j] == None]
        top_left = choices[0]
        min_val = choices[0][0] + choices[0][1]

        for choice in choices[1:]:
            if choice[0] + choice[1] < min_val:
                top_left = choice
                min_val = choice[0] + choice[1]

        return top_left