class TreePlayerHeuristic :
    def __init__(self, layers) :
        self.num = None
        self.tree = None
        self.layers = layers

    def heuristic(self, board) :
        rows = [board[row] for row in range(6)] #row
        cols = [''.join([board[row][col] for row in range(6)]) for col in range(7)]
        l_dias = []
        r_dias = []
        diagonals = [(3,0),(4,0),(5,0),(5,1),(5,2),(5,3)]
        for (row, col) in diagonals :
            i=0
            l_dia = []
            r_dia = []
            while row-i >=0 and col+i <= 6 :
                l_dia.append(board[row-i][col+i])
                r_dia.append(board[row-i][6-col-i])
                i+= 1
            l_dias.append(l_dia)
            r_dias.append(r_dia)
        
        win_process = rows + cols + l_dias + r_dias

        good_set = 0
        bad_set = 0
        for thing in win_process :
            good_num = thing.count(str(1))
            bad_num = thing.count(str(2))
            empty = thing.count('0')
            if good_num >= 2 and empty >= 2 :
                good_set+= 1
            if bad_num >= 2 and empty >= 2 :
                bad_set+=1
        return (good_set-bad_set)/len(win_process)
  
    def set_player_number(self, n) :
        self.num = n
        self.tree = Con4Tree(n, self.layers, self.heuristic)

    def choose_move(self, board, choices) :
        self.tree.prune_tree(board)
        best = (None, -1)
        #print(self.tree.nodes.keys())

        for col in choices :
            update = board.copy()
            choice = self.find_row(board, col)
            update[choice[0]] = board[choice[0]][:choice[1]] + str(self.num) + board[choice[0]][choice[1]+1:]
            flatdate = self.tree.flaten_board(update)
            if self.tree.nodes[flatdate].score >= best[1] :
                best = (choice, self.tree.nodes[flatdate].score)
        return best[0]

    def find_row(self, board, col) :
        for row in range(5,-1,-1) :
            if board[row][col] == '0' :
                return (row, col)

class Node :
    def __init__(self, parent, player, game_state) :
        self.children = []
        self.player = player
        self.parent = [parent]
        self.score = None
        self.winner = self.check_for_winner(game_state)
    
    def check_for_winner(self, board) :
        rows = [board[row] for row in range(6)] #row
        cols = [''.join([board[row][col] for row in range(6)]) for col in range(7)]
        '''
        print(board)
        for col in range(7) :
            for row in range(6) :
                print(board[row][col], (row,col))
        #'''
        l_dias = []
        r_dias = []
        diagonals = [(3,0),(4,0),(5,0),(5,1),(5,2),(5,3)]
        for (row, col) in diagonals :
            i=0
            l_dia = []
            r_dia = []
            while row-i >=0 and col+i <= 6 :
                l_dia.append(board[row-i][col+i])
                r_dia.append(board[row-i][6-col-i])
                i+= 1
            l_dias.append(l_dia)
            r_dias.append(r_dia)
        
        thing = rows + cols + l_dias + r_dias

        tie = []

        for stuff in thing :
            tie.append('0' in stuff)
            if stuff.count('0') > len(stuff) - 4 :
                continue
            if '1111' in stuff :
                return 1
            elif '2222' in stuff :
                return 2

        if True not in tie:
            return 'Tie'

class Con4Tree :
    def __init__(self, max_plr, num_layers, heuristic_funct) :
        self.root = None
        self.layer_num = num_layers
        self.heuristic = heuristic_funct
        self.max_plr = max_plr
        self.leaf_nodes = [] 
        self.nodes = {}
    
    def find_turn(self, board) : 
        if board.count('1') == board.count('2') :
            return 1
        else :
            return 2

    def get_possible_moves(self, game_state) :
        if self.nodes[game_state].winner != None :
            return [[]]

        board = self.inflate_board(game_state)
        
        possible_moves = []
        
        cols = [[(row,col) for row in range(5,-1,-1)] for col in range(7)]
        for col in cols :
            for (row, col) in col :
                if board[row][col] == '0' :
                    possible_moves.append((row,col))
                    break
        return possible_moves     
    
    def create_nodes(self, start_board) :
        prev_choices = [start_board]
        curr_plr = self.find_turn(start_board)
        layer = 0
        while layer < self.layer_num and prev_choices != [] :
            choice = prev_choices[0]
            if self.find_turn(choice) != curr_plr :
                layer += 1
                curr_plr = self.find_turn(choice)
            if layer == self.layer_num :
                break
            prev_choices.remove(choice)

            possible_choices = self.get_possible_moves(choice)
            if [] in possible_choices :
                self.leaf_nodes.append(choice)
                continue

            if choice in self.nodes :
                prev_choices.extend(self.nodes[choice].children)
                if self.nodes[choice].children != [] :
                    continue

            update = []
            for (row, col) in possible_choices :
                new = self.inflate_board(choice)
                new[row] = new[row][:col] + str(curr_plr) + new[row][col+1:]
                update.append(new)

            for board in update :
                move = self.flaten_board(board)
                if move in self.nodes.keys() :
                    self.nodes[move].parent.append(choice)
                else :
                    prev_choices.append(move)
                    self.nodes[move] = Node(choice, (curr_plr%2) +1, board)
            
        self.leaf_nodes.extend(prev_choices)
        if len(set(self.leaf_nodes)) != len(self.leaf_nodes) :
            print("ERROR")

    def assign_values(self) :
        unassigned = self.leaf_nodes.copy()
        index = 0

        while len(unassigned) >= 1  :
            if index == len(unassigned) and unassigned != [] :
                index = 0

            node = self.nodes[unassigned[index]]
            unassigned.extend([parent for parent in node.parent if self.nodes[parent].score != 'root' and parent not in unassigned])

            child_scores = [self.nodes[child].score for child in node.children]
            if None in child_scores :
                index += 1
                continue
            
            if child_scores == [] :
                self.find_score(unassigned[index], node)
            elif node.player == self.max_plr :
                node.score = max(child_scores)
            else :
                node.score = min(child_scores)
            unassigned.pop(index)
    
    def find_score(self, board, node) :
        if node.winner != None :
            if node.winner == 'Tie' :
                node.score = 0
            elif node.winner == self.max_plr :
                node.score = 1
            else :
                node.score = -1
            return
        
        node.score = self.heuristic(self.inflate_board(board))

    def prune_tree(self, new_board) :
        self.leaf_nodes = []
        self.root = self.flaten_board(new_board)
        if self.root not in self.nodes.keys() :
            self.nodes = {self.root: Node(None, self.find_turn(self.root), new_board)}
        self.nodes[self.root].score = 'root'
        self.create_nodes(self.root)
        self.assign_values()

    def flaten_board(self, board) :
        return ''.join(board)

    def inflate_board(self, board) :
        return [board[index:index+7] for index in range(0,41,7)]