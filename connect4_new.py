import random
import copy

class ConnectFourAI:
    def __init__(self, game):
        self.game = game

    def copy_game(self):
        new_game = ConnectFour()
        new_game.board = [row.copy() for row in self.game.board]
        new_game.bitboards = self.game.bitboards.copy()
        new_game.player_turn = self.game.player_turn
        return new_game
    
    def save_heuristics_to_file(heuristics, filename):
        with open(filename, 'a') as f:  # 'a' mode appends to the file instead of overwriting it
            for heuristic, score in heuristics.items():
                f.write(f"{heuristic}: {score}\n")
            f.write("\n")  # Add a newline to separate different sets of heuristics


    def evaluate(self, game):
        print("Evaluating game state:\n", game.board)
        """Return the heuristic evaluation of the current game state."""
        
        score = 0

        # 1. Immediate win
        score += self.immediate_win(game)
        
        # 3. Blocking opponent's Connect Fours
        score += self.block_opponent_connect_fours(game)
        print("score after blocking opponent's connect fours: ", score)

        score += self.block_opponent_connect_three(game)
        print("score after blocking opponent's connect threes: ", score)

        # 2. Create potential connect fours
        score += self.create_potential_connect_fours(game)
        print("score after creating potential connect fours: ", score)

        score += self.create_potential_connect_three(game)
        print("score after creating potential connect threes: ", score)

        # 3. Center column preference
        score += self.center_column_preference(game)
        print("score after center column preference: ", score)

        # 4. Creating multiple threats
        score += self.create_multiple_threats(game)
        print("score after creating multiple threats: ", score)

        # 5. Odd and even rows
        score += self.odd_even_rows(game)
        print("score after odd and even rows: ", score)

        # 6. Avoiding trap states
        score += self.avoid_trap_states(game)
        print("score after avoiding trap states: ", score)

        
        """ 
        print("Potential connect fours: ", self.create_potential_connect_fours(game))
        print("Center column preference: ", self.center_column_preference(game))
        print("Blocking opponent's connect fours: ", self.block_opponent_connect_fours(game))
        print("Creating multiple threats: ", self.create_multiple_threats(game))
        print("Odd and even rows: ", self.odd_even_rows(game))
        print("Avoiding trap states: ", self.avoid_trap_states(game)) 
        """

        return score

    def create_potential_connect_fours(self, game):
        score = 0
        bitboard = game.bitboards[game.player_turn - 1]  # Get the bitboard for the current player

        # Define the four directions as bit shifts
        directions = [1, 7, 6, 8]

        for direction in directions:
            # Shift the bitboard in the current direction to get the positions of the pieces
            # if they were to move one step in this direction
            shifted_bitboard = bitboard >> direction

            # Count the number of potential Connect Fours by performing a bitwise AND operation
            # between the bitboard and the shifted bitboard. This will give a new bitboard where
            # a bit is set to 1 only if the corresponding bit is set to 1 in both the original
            # bitboard and the shifted bitboard, i.e., if moving a piece one step in the current
            # direction would create a Connect Four.
            potential_connect_fours = bitboard & shifted_bitboard

            # Count the number of bits set to 1 in the potential_connect_fours bitboard
            # This is equivalent to counting the number of potential Connect Fours
            consecutive_pieces = bin(potential_connect_fours).count('1')

            # Add the score for the potential Connect Fours to the total score
            score += consecutive_pieces * 40

        return score
    
    def create_potential_connect_three(self, game):
        score = 0
        bitboard = game.bitboards[game.player_turn - 1]  # Get the bitboard for the current player

        # Define the three directions as bit shifts
        directions = [1, 7, 6]

        for direction in directions:
            # Shift the bitboard in the current direction to get the positions of the pieces
            # if they were to move one step in this direction
            shifted_bitboard = bitboard >> direction

            # Count the number of potential Connect Threes by performing a bitwise AND operation
            # between the bitboard and the shifted bitboard. This will give a new bitboard where
            # a bit is set to 1 only if the corresponding bit is set to 1 in both the original
            # bitboard and the shifted bitboard, i.e., if moving a piece one step in the current
            # direction would create a Connect Three.
            potential_connect_threes = bitboard & shifted_bitboard

            # Count the number of bits set to 1 in the potential_connect_threes bitboard
            # This is equivalent to counting the number of potential Connect Threes
            consecutive_pieces = bin(potential_connect_threes).count('1')

            # Add the score for the potential Connect Threes to the total score
            score += consecutive_pieces * 30

        return score
    
    def center_column_preference(self, game):
        # Define a mask for the center column
        center_column_mask = 0b1000000100000010000001000000100000100000

        # Apply the mask to the bitboards of the two players
        center_column_player = game.bitboards[game.player_turn - 1] & center_column_mask
        center_column_opponent = game.bitboards[2 - game.player_turn] & center_column_mask

        # Count the number of pieces each player has in the center column
        center_count_player = bin(center_column_player).count('1')
        center_count_opponent = bin(center_column_opponent).count('1')

        print("center count player: ", center_count_player)
        print("center count opponent: ", center_count_opponent)
        print("After center column preference, score:", center_count_player * 40 - center_count_opponent * 40)
        print("center count player * 10 - center count opponent * 10: ", center_count_player * 40 - center_count_opponent * 40)

        return center_count_player * 40 - center_count_opponent * 40

    def block_opponent_connect_fours(self, game):
        score = 0
        for column in range(7):
            for row in range(6):
                if (game.bitboards[2 - game.player_turn] & (1 << (row * 7 + column))) == 0:
                    game.bitboards[2 - game.player_turn] |= (1 << (row * 7 + column))
                    if self.check_win_bitboard(game.bitboards[2 - game.player_turn]):
                        print("Blocking opponent's connect four at column ", column, " row ", row)
                        score -= 100000000
                    game.bitboards[2 - game.player_turn] &= ~(1 << (row * 7 + column))
                    break
        print("After blocking opponent's connect fours, score:", score)
        return score
    
    def block_opponent_connect_three(self, game):
        score = 0
        for column in range(7):
            for row in range(6):
                if (game.bitboards[2 - game.player_turn] & (1 << (row * 7 + column))) == 0:
                    game.bitboards[2 - game.player_turn] |= (1 << (row * 7 + column))
                    if self.create_potential_connect_three(game) > 0:
                        score -= 100000000
                    game.bitboards[2 - game.player_turn] &= ~(1 << (row * 7 + column))
                    break
        return score

    def check_win_bitboard(self, bitboard):
        # Horizontal check
        m = bitboard & (bitboard >> 7)
        if m & (m >> 14):
            return True
        # Diagonal \ check
        m = bitboard & (bitboard >> 6)
        if m & (m >> 12):
            return True
        # Diagonal / check
        m = bitboard & (bitboard >> 8)
        if m & (m >> 16):
            return True
        # Vertical check
        m = bitboard & (bitboard >> 1)
        if m & (m >> 2):
            return True
        # No win
        return False

    def create_multiple_threats(self, game):
        score = 0
        bitboard = game.bitboards[game.player_turn - 1]
        print("\n")
        print("Player: ", game.player_turn)
        print("Move: ", bin(bitboard))

        # Horizontal threats
        m = bitboard & (bitboard >> 7)
        score += 5 * bin(m & (m >> 14) & ~(bitboard >> 21)).count('1')

        # Vertical threats
        m = bitboard & (bitboard >> 1)
        score += 5 * bin(m & (m >> 2) & ~(bitboard >> 3)).count('1')

        # Diagonal \ threats
        m = bitboard & (bitboard >> 6)
        score += 5 * bin(m & (m >> 12) & ~(bitboard >> 18)).count('1')

        # Diagonal / threats
        m = bitboard & (bitboard >> 8)
        score += 5 * bin(m & (m >> 16) & ~(bitboard >> 24)).count('1')

        print("After creating multiple threats, score:", score)
        return score

    def odd_even_rows(self, game):
        bitboard = game.bitboards[0]  # Player 1's bitboard

        # Masks for even rows
        masks = [0b1111111 << (7 * i) for i in range(0, 6, 2)]

        score = sum(bin(bitboard & mask).count('1') for mask in masks)

        print("After odd and even rows, score:", score)
        return score

    def avoid_trap_states(self, game):
        score = 0
        for column in range(7):
            for row in reversed(range(6)):
                if game.board[row][column] == 0:
                    # Simulate placing a piece in this cell
                    game.board[row][column] = game.player_turn
                    game.bitboards[game.player_turn - 1] |= (1 << (row * 7 + column))
                    # Check if this creates a trap state
                    if self.creates_trap_state(game, column, row):
                        # If it does, subtract a large value from the score for this column
                        score -= 100000000
                    # Remove the piece from the cell
                    game.board[row][column] = 0
                    game.bitboards[game.player_turn - 1] &= ~(1 << (row * 7 + column))
                    break
        print("After avoiding trap states, score:", score)
        return score

    def creates_trap_state(self, game, column, row):
        # Convert the column and row to a single index
        index = row * 7 + column

        # Create masks for the four directions
        vertical_mask = 0b1000100010001 << index
        horizontal_mask = 0b10001 << index
        positive_diagonal_mask = 0b1000001000001 << index
        negative_diagonal_mask = 0b1000100001000001 << index

        # Check for a trap state in each direction
        for mask in [vertical_mask, horizontal_mask, positive_diagonal_mask, negative_diagonal_mask]:
            if (game.bitboards[2 - game.player_turn] & mask) == mask:
                print("Trap state found")
                return True

        print("No trap state found")
        return False
        
    def immediate_win(self, game):
        """Return a high score if the AI can win in the next move."""
        for column in range(7):
            for row in range(6):
                if game.board[row][column] == 0:
                    # Simulate placing the AI's piece in this cell
                    game.board[row][column] = self.game.player_turn
                    game.bitboards[self.game.player_turn - 1] |= (1 << (row * 7 + column))
                    # If this move would cause the AI to win
                    if self.check_win_bitboard(game.bitboards[self.game.player_turn - 1]):
                        # Remove the AI's piece from the cell
                        game.board[row][column] = 0
                        game.bitboards[self.game.player_turn - 1] &= ~(1 << (row * 7 + column))
                        # Return a high score
                        return 100000000
                    # Remove the AI's piece from the cell
                    game.board[row][column] = 0
                    game.bitboards[self.game.player_turn - 1] &= ~(1 << (row * 7 + column))
                    break
        # If the AI can't win in the next move, return 0
        return 0

    def is_full(self, game):
        full_board = 0b1111111111111111111111111111111111111111111  # Bitboard representing a full board
        return (game.bitboards[0] | game.bitboards[1]) == full_board

    def minimax(self, game_copy, depth, alpha, beta, maximizing_player):
        """Return the minimax value of the current game state."""

        #check if board is full
        if self.is_full(game_copy):
            return 0

        # Base case: leaf node
        if depth == 0:
            eval_score = self.evaluate(game_copy)
            print(f"Depth: {depth}, Eval Score: {eval_score}")
            return eval_score

        # Recursive case: maximize your gains or minimize the opponent's gains
        value = -float('inf') if maximizing_player else float('inf')
        columns = list(range(7))
        random.shuffle(columns)  # Randomize the order of the columns
        print(f"Columns Order: {columns}")

        if maximizing_player:
            for column in columns:
                if game_copy.board[0][column] == 0:
                    game_copy.make_move(column)
                    value = max(value, self.minimax(game_copy, depth - 1, alpha, beta, False))
                    print(f"Maximizing, Depth: {depth}, Column: {column}, Value: {value}, Alpha: {alpha}, Beta: {beta}")
                    game_copy.undo_move(column)
                    alpha = max(alpha, value)
                    if alpha >= beta:
                        print("Alpha-beta pruning in maximizing player")
                        break
        else:
            for column in columns:
                if game_copy.board[0][column] == 0:
                    game_copy.make_move(column)
                    value = min(value, self.minimax(game_copy, depth - 1, alpha, beta, True))
                    print(f"Minimizing, Depth: {depth}, Column: {column}, Value: {value}, Alpha: {alpha}, Beta: {beta}")
                    game_copy.undo_move(column)
                    beta = min(beta, value)
                    if alpha >= beta:
                        print("Alpha-beta pruning in minimizing player")
                        break
        return value
    
    def is_column_full(self, column, game):
        top_bit = 1 << (5 * 7 + column)  # Bit representing the top cell of the column
        return (game.bitboards[0] | game.bitboards[1]) & top_bit != 0

    def get_move(self):
        print("get_move called")
        print("Game state before AI move:")
        self.game.print_board()
        game_copy = copy.deepcopy(self.game)
        # Create a deep copy of the game board
        original_board = copy.deepcopy(self.game.board)
        # Iterate over each column
        for column in range(7):
            # If the column is full, skip it
            if self.is_column_full(column, self.game):
                continue
            # Find the next available row in this column
            for row in reversed(range(6)):
                if game_copy.board[row][column] == 0:
                    # Temporarily place the opponent's piece in the cell
                    game_copy.board[row][column] = 3 - self.game.player_turn
                    # If this move would cause the opponent to win
                    if self.game.check_win(3 - self.game.player_turn):
                        # Remove the opponent's piece from the cell
                        game_copy.board[row][column] = 0
                        # Return this column as the move to make
                        return column
                
       # If no move was found that blocks the opponent's win or creates a trap state, use minimax to determine the best move
        best_score = -float('inf')
        best_move = -1
        for column in range(7):
            if self.game.board[0][column] == 0:
                game_copy = copy.deepcopy(self.game)
                game_copy.make_move(column)
                score = self.minimax(game_copy, 7, -float('inf'), float('inf'), True)  # Adjust depth as needed
                print("column ", column, " score: ", score)
                if score > best_score:
                    best_score = score
                    best_move = column
                    print("best score: ", best_score)
                    print("best move: ", best_move)
        print("best move: ", best_move)
        print("Game state after AI move:")
        self.game.print_board()
        return best_move

class ConnectFour:
    def __init__(self):
        self.board = [[0 for _ in range(7)] for _ in range(6)]
        self.bitboards = [0, 0]  # Bitboards for the two players
        self.player_turn = random.choice([0, 1]) + 1  # Randomly choose which player goes first

    def make_move(self, column):
        if column < 0 or column > 6:
            print("Invalid move: column must be between 0 and 6.")
            return None
        if self.board[0][column] != 0:
            print("Invalid move: column is full.")
            return None
        for row in reversed(range(6)):
            if self.board[row][column] == 0:
                self.board[row][column] = self.player_turn
                self.bitboards[self.player_turn - 1] |= (1 << (row * 7 + column))
                if self.check_win(self.player_turn):
                    return True
                return False
        print("Invalid move: column is full.")
        return None

    def undo_move(self, column):
        """Undo the last move made in the given column."""
        for row in range(6):
            if self.board[row][column] != 0:
                self.board[row][column] = 0
                self.bitboards[self.player_turn - 1] &= ~(1 << (row * 7 + column))
                self.player_turn = 3 - self.player_turn
                break

    def check_win(self, player):
        # Check horizontal locations for win
        for c in range(4):
            for r in range(6):
                if self.board[r][c] == player and self.board[r][c+1] == player and self.board[r][c+2] == player and self.board[r][c+3] == player:
                    return True

        # Check vertical locations for win
        for c in range(7):
            for r in range(3):
                if self.board[r][c] == player and self.board[r+1][c] == player and self.board[r+2][c] == player and self.board[r+3][c] == player:
                    return True

        # Check positively sloped diagonals
        for c in range(4):
            for r in range(3):
                if self.board[r][c] == player and self.board[r+1][c+1] == player and self.board[r+2][c+2] == player and self.board[r+3][c+3] == player:
                    return True

        # Check negatively sloped diagonals
        for c in range(4):
            for r in range(3, 6):
                if self.board[r][c] == player and self.board[r-1][c+1] == player and self.board[r-2][c+2] == player and self.board[r-3][c+3] == player:
                    return True

        return False

    def is_full(self):
        return all(self.board[0][column] != 0 for column in range(7))

    def print_board(self):
        print('\n'.join([' '.join(map(str, row)) for row in self.board]))

    def play_game(self):
        turn = 0
        ai = ConnectFourAI(self)
        while not self.is_full():
            turn += 1
            print("\n")
            print(f"Turn {turn}")
            print(f"Player {self.player_turn}'s turn:")
            self.print_board()
            if self.player_turn == 1:
                column = int(input(f"Player {self.player_turn}, choose a column: ")) - 1
                win = self.make_move(column)
                if win is True:
                    print(f"Player {self.player_turn} wins!")
                    self.print_board()
                    return
                elif win is None:
                    continue
                self.player_turn = 3 - self.player_turn
            else:
                if turn == 2 or turn == 1:
                    column = 3
                else:
                    column = ai.get_move()
                win = self.make_move(column)
                if win is True:
                    print(f"Player {self.player_turn} wins!")
                    self.print_board()
                    return
                elif win is None:
                    continue
                self.player_turn = 3 - self.player_turn
        print("The game is a draw.")

def main():
    game = ConnectFour()
    game.play_game()

if __name__ == '__main__':
    main()