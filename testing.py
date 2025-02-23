#visit https://lichess.org/api/games/user/tumnut

from collections import defaultdict

# Initialize counter
total_counter = 0
white_counter = 0
fourteen_counter = 0

all_white_games_first_ten_moves = []
two_moves = defaultdict(int)
three_moves = defaultdict(int)
four_moves = defaultdict(int)

# Open the file in read mode
with open('C:\\Users\\Thomas\\Downloads\\lichess_tumnut_2025-01-20.pgn', 'r') as file:
    # Iterate through each line in the file
    for line in file:
        if fourteen_counter == 14:
            split_line = line.split(" ")
            del split_line[0::3]
            # print(split_line[0:10])
            if len(split_line) > 0 and split_line[0] == 'd4':# and split_line[2] != 'Bf4':
                all_white_games_first_ten_moves.append(split_line[0:10])
                two_moves[" ".join(split_line[0:4])] += 1
                three_moves[" ".join(split_line[0:6])] += 1
                four_moves[" ".join(split_line[0:8])] += 1
            fourteen_counter = 0
        if fourteen_counter > 0:
            fourteen_counter += 1
            continue
        # Remove any extra spaces or newline characters
        line = line.strip()
        if '[Event "Rated blitz' in line or '[Event "Rated rapid' in line or '[Event "Rated classical' in line:
            total_counter += 1
        if line == '[White "tumnut"]':
            # Increment the counter
            white_counter += 1
            # Then count 14 lines down, and that should be the notation
            fourteen_counter = 1

# Print the final count
print(f'tumnut played white {white_counter} times out of {total_counter} games.')
# print(two_moves.items())
print(dict(sorted([it for it in two_moves.items() if it[1] > 5], key=lambda item: -1*item[1])))
# print(dict(sorted([it for it in three_moves.items() if it[1] > 4], key=lambda item: -1*item[1])))
# print(dict(sorted([it for it in four_moves.items() if it[1] > 3], key=lambda item: -1*item[1])))\

# for b_i, b in enumerate([1, 2, 3, 4, 5, 6, 7, 8]):
#     for a_i, a in enumerate(['h', 'g', 'f', 'e', 'd', 'c', 'b', 'a']):
#         for d_i, d in enumerate([1, 8]):
#             for c_i, c in enumerate(['h', 'g', 'f', 'e', 'd', 'c', 'b', 'a']):
#                 for e_i, e in enumerate(['b', 'n', 'r', 'q']):
#                     one = a + str(b)
#                     two = c + str(d)
#                     if abs(a_i-c_i) > 1 or abs(b-d) != 1:
#                         continue
#                     print(one+two+e, end='')