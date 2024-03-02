from PIL import Image
import matplotlib.pyplot as plt
import math
import numpy as np
import heapq
import copy
import time

n = 4
home_dict = {}
image_index_dict = {}
cost_dict = {}

class Rect():
    def __init__ (self, numbering, r, c):
        self.numbering = numbering
        self.current_r = r
        self.current_c = c
    def __str__(self):
        return str(self.numbering)
    def get_h(self):
        home_r,home_c = home_dict[self.numbering]
        return max(abs(self.current_r-home_r),abs(self.current_c-home_c))
    
    
grid = [Rect((str(i) if i < n*n-1 else " "),int(i/n),i%n) for i in range(n*n)]
blank_row = n-1
blank_col = n-1
h_beginning = 0
    
def grid_to_string(in_grid):
    s = []
    for i in range(n):
        s.append("".join([f"[{f"{str(j)}".center((3 + math.ceil(math.log(10,n*n))), " ")}]" for j in in_grid[i*n:int(i*n) + n]]))
    return ("\n".join([i for i in s]))
    
def print_grid(to_print):
    print(grid_to_string(to_print))
    
def calculate_h(in_grid):
    return sum([(i.get_h() if i.numbering != " " else 0) for i in in_grid])
    
def shuffle_grid():
    global grid
    global blank_row
    global blank_col
    
    a = blank_row
    b = blank_col
    applications = [apply_up,apply_down,apply_left,apply_right]
    
    for i in range(n*n*n):
        r = np.random.randint(4)
        
        if (r == 0 and a > 0) or (r == 1 and a < n-1) or (r == 2 and b > 0) or (r == 3 and b < n-1):
            a,b,grid = (applications[r])(a,b,grid)
    
    blank_row = a
    blank_col = b
    return a, b
    
def apply_up(b_r,b_c,grid):
    rv = copy.deepcopy(grid)
    temp = copy.deepcopy(rv[(b_r-1)*n+b_c])
    rv[(b_r-1)*n+b_c] = copy.deepcopy(rv[b_r*n+b_c])
    rv[b_r*n+b_c] = temp
    
    rv[(b_r-1)*n+b_c].current_r = b_r-1
    rv[(b_r-1)*n+b_c].current_c = b_c
    
    rv[b_r*n+b_c].current_r = b_r
    rv[b_r*n+b_c].current_c = b_c

    return b_r-1,b_c,rv

def apply_down(b_r,b_c,grid):
    rv = copy.deepcopy(grid)
    temp = copy.deepcopy(rv[(b_r+1)*n+b_c])
    rv[(b_r+1)*n+b_c] = copy.deepcopy(rv[b_r*n+b_c])
    rv[b_r*n+b_c] = temp
    
    rv[(b_r+1)*n+b_c].current_r = b_r+1
    rv[(b_r+1)*n+b_c].current_c = b_c
    
    rv[b_r*n+b_c].current_r = b_r
    rv[b_r*n+b_c].current_c = b_c
    
    return b_r+1,b_c,rv

def apply_left(b_r,b_c,grid):
    rv = copy.deepcopy(grid)
    temp = copy.deepcopy(rv[b_r*n+b_c-1])
    rv[b_r*n+b_c-1] = copy.deepcopy(rv[b_r*n+b_c])
    rv[b_r*n+b_c] = temp
    
    rv[b_r*n+b_c-1].current_r = b_r
    rv[b_r*n+b_c-1].current_c = b_c-1
    
    rv[b_r*n+b_c].current_r = b_r
    rv[b_r*n+b_c].current_c = b_c
    
    
    return b_r,b_c-1,rv

def apply_right(b_r,b_c,grid):
    rv = copy.deepcopy(grid)
    temp = copy.deepcopy(rv[b_r*n+b_c+1])
    rv[b_r*n+b_c+1] = copy.deepcopy(rv[b_r*n+b_c])
    rv[b_r*n+b_c] = temp
    
    rv[b_r*n+b_c+1].current_r = b_r
    rv[b_r*n+b_c+1].current_c = b_c+1
    
    rv[b_r*n+b_c].current_r = b_r
    rv[b_r*n+b_c].current_c = b_c

    return b_r,b_c+1,rv

def stringify(to_string):
    return "".join([str(i) for i in to_string])

class Grid():
    def __init__(self, in_grid = [], moves = [], blank_r = 0, blank_c = 0, g = 0, h = 0):
        self.grid = in_grid
        self.moves = moves
        self.blank_r = blank_r
        self.blank_c = blank_c

        self.g = g
        self.h = h
        
    def __str__(self):
        return grid_to_string(self.grid)
    def f(self):
        return self.g + self.h
    def get_neighbors(self):
        gamma = []
        
        if self.blank_r > 0:
            a,b,up_grid = apply_up(self.blank_r,self.blank_c,self.grid)

            gamma.append(Grid(in_grid=up_grid,moves = self.moves+['u'],blank_r=a,blank_c=b,g=self.g+1,h=calculate_h(up_grid)))
        
        if self.blank_r < n-1:
            a,b,down_grid = apply_down(self.blank_r,self.blank_c,self.grid)
            gamma.append(Grid(in_grid=down_grid,moves = self.moves+['d'],blank_r=a,blank_c=b,g=self.g+1,h=calculate_h(down_grid)))
            
        if self.blank_c > 0:
            a,b,left_grid = apply_left(self.blank_r,self.blank_c,self.grid)
            gamma.append(Grid(in_grid=left_grid,moves = self.moves+['l'],blank_r=a,blank_c=b,g=self.g+1,h=calculate_h(left_grid)))
            
        if self.blank_c < n-1:  
            a,b,right_grid = apply_right(self.blank_r,self.blank_c,self.grid)
            gamma.append(Grid(in_grid=right_grid,moves = self.moves+['r'],blank_r=a,blank_c=b,g=self.g+1,h=calculate_h(right_grid)))
            
        return gamma
    
    def is_solved(self):
        return (self.h == 0)
    def __eq__(self, other):
        return self.f() == other.f()
    def __ge__(self, other):
        return self.f() > other.f()
    def __hash__(self) -> int:
        x = stringify(self.grid)
        return hash(x)
    
def plot_grid(grid,axes,subplot_dim,fig):
    for i in range(n*n):
        axis = axes[i]
        image_index = image_index_dict[grid[i].numbering]
        subimg = img[int(image_index/n)*subplot_dim:(int(image_index/n)*subplot_dim+subplot_dim),(image_index%n)*subplot_dim:((image_index%n)*subplot_dim+subplot_dim),:]
        if grid[i].numbering != " ":
            axes[i].imshow(subimg)
        else:
            axes[i].imshow(np.full(shape=(subplot_dim,subplot_dim,3),fill_value=255))

    plt.pause(0.1)
        
if __name__ == "__main__":    
    for i,rect in enumerate(grid):
        image_index_dict[rect.numbering] = i
        home_dict[rect.numbering] = (rect.current_r,rect.current_c)

    # time.sleep(5)

    _, _ = shuffle_grid()
    print_grid(grid)
    
    initial = Grid(grid,[],blank_row,blank_col,0,calculate_h(grid))
    print("\n")
    
    # print(blank_row,blank_col)
    q = [(initial.f(),initial)]
    
    x = None
    
    visited = set([stringify(initial.grid)])
    cost_dict[stringify(initial.grid)] = initial.g

    while q:
        x = heapq.heappop(q)[1]
        if x.is_solved():
            print("FOUND SOLUTION")
            break
        
        neighbors = x.get_neighbors()
                
        for neighbor in neighbors:
            if stringify(neighbor.grid) not in visited:
                visited.add(stringify(neighbor.grid))
                cost_dict[stringify(neighbor.grid)] = neighbor.g
                heapq.heappush(q,(neighbor.f(),neighbor))
            elif cost_dict[stringify(neighbor.grid)] > neighbor.g:
                cost_dict[stringify(neighbor.grid)] = neighbor.g
                heapq.heappush(q,(neighbor.f(),neighbor))
    
    moves = x.moves
    
    a = blank_row
    b = blank_col
    
    img = plt.imread('turing.jpg')
    min_dim = min(img.shape[:2])
    min_dim = int(min_dim - (min_dim%n))
    img = img[:min_dim,:min_dim,:]
    subplot_dim = int(min_dim/n)
    
    fig = plt.figure(figsize=(5,5)) # specifying the overall grid size
    axes = [plt.subplot(n,n,i+1) for i in range(n*n)]
    
    for i in range(n*n):
        axes[i].set_xticklabels([])
        axes[i].set_yticklabels([])
        # axes[i].axis("off")
    
    fig.subplots_adjust(wspace=0, hspace=0)

    plt.ion()
    plt.show()
    
    plot_grid(grid,axes,subplot_dim,fig)
    
    move_to_app = {'u':apply_up,'d':apply_down,'l':apply_left,'r':apply_right}
    
    for move in moves:
        a,b,grid = move_to_app[move](a,b,grid)
        plot_grid(grid,axes,subplot_dim,fig)
    
    plt.waitforbuttonpress()