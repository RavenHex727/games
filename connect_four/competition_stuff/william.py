class SmartRandomPlayer:
    def __init__(self):
        self.player_num = None
        self.invalid_moves = 0

    def set_player_number(self, n):
        self.player_num = n

    def choose_move(self, board, choices):
        player_win = False
        opp_win = False
        for space in self.find_open_spaces(board):
            winner = self.check_winner(self.update_board(space, board))
            if winner == self.player_num:
                player_win = space
            elif winner == 3 - self.player_num:
                opp_win = space

        if player_win: return player_win
        if opp_win: return opp_win
        return random.choice(self.find_open_spaces(board))

    def check_winner(self, state):
        rows = self.four_in_list(state)
        if rows:
            return rows

        cols = self.four_in_list([''.join(i) for i in zip(*state)])
        if cols:
            return cols

        fdiag = ['' for _ in range(len(state) + len(state[0]) - 1)]
        bdiag = ['' for _ in range(len(fdiag))]
        for x in range(len(state[0])):
            for y in range(len(state)):
                fdiag[x + y] += state[y][x]
                bdiag[x - y - (1 - len(state))] += state[y][x]
        diags = fdiag + bdiag
        diag = self.four_in_list(diags)
        if diag:
            return diag
        
        return False

    def update_board(self, move, board):
        board = board.copy()
        cols = [''.join(i) for i in zip(*board)]
        chosen_col = [*cols[move]]

        col_idx = len(chosen_col)-1 - list(reversed(chosen_col)).index('0')
        row = board[col_idx]
        board[col_idx] = row[:move] + str(self.player_num) + row[move+1:]
        return board

    def four_in_list(self, lst):
        for string in lst:
            for i in range(0, len(string)-3):
                if string[i] == string[i+1] == string[i+2] == string[i+3] != '0':
                    return string[i]
        return False

    def find_open_spaces(self, board):
        moves = []
        t_board = [''.join(i) for i in zip(*board)]
        for i, col in enumerate(t_board):
            if '0' in col:
                moves.append(i)
        return moves