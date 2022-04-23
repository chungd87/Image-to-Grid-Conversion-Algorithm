rows = 11
columns = 11

grid = [[" " for x in range(rows)] for x in range(columns)]

for i in range(0,rows):
    grid[i][0] = '#'
    grid[i][rows-1] = '#'
    grid[0][i] = '#'
    grid[columns-1][i] = '#'

grid[1][2] = '#'
grid[2][2] = '#'
grid[2][4] = '#'
grid[2][5] = '#'
grid[2][6] = '#'
grid[2][7] = '#'
grid[2][8] = '#'
grid[2][9] = '#'
grid[3][2] = '#'
grid[4][2] = '#'
grid[4][3] = '#'
grid[4][4] = '#'
grid[4][5] = '#'
grid[4][6] = '#'
grid[4][7] = '#'
grid[4][8] = '#'
grid[6][1] = '#'
grid[6][2] = '#'
grid[6][3] = '#'
grid[6][4] = '#'
grid[6][5] = '#'
grid[6][6] = '#'
grid[6][8] = '#'
grid[6][9] = '#'
grid[8][2] = '#'
grid[8][3] = '#'
grid[8][4] = '#'
grid[8][5] = '#'
grid[8][6] = '#'
grid[8][7] = '#'
grid[8][8] = '#'
grid[8][9] = '#'
grid[9][4] = '#'





for line in grid:
    print(line)
