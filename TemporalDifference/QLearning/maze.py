import pandas as pd
import numpy as np

class Maze:
    
    def __init__(self,goal=[3,3],trap1=[0,3],trap2=[3,1],position=0):
        pass
        
    '''
    def printTable(self,p=None):
        p = random_position()
        table = pd.DataFrame(np.zeros((4,4),dtype=int),columns=None)
        table.iloc[3,3]='X'
        table.iloc[3,1]='T'
        table.iloc[0,3]='T'
        T = pd.DataFrame({
              'linhas':[0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3],\
              'colunas':[0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]
                          })
        table = table.replace(0,'_')
        table.iloc[T['linhas'][p],T['colunas'][p]] = 'o'
        print(table.to_string(index=False,header=False))
        print('')
    '''
    @property
    def random_position(self):
        return np.random.randint(0,16)
    
    def set_traps(self,trap1,trap2):
        self.trap1 = trap1
        self.trap2 = trap2
    
    def set_goal(self,goal):
        self.goal = goal
    
    def get_goal(self):
        return self.goal
    
    def set_position(self,position):
        self.position = position
        
    def get_posisiotn(self):
        return self.position
    
    def make_maze(self,goal,trap1,trap2,position):
        self.table = pd.DataFrame(np.zeros((4,4),dtype=int),columns=None)
        self.table.iloc[goal[0],goal[1]]='X'
        self.table.iloc[self.trap1[0],self.trap1[1]]='T'
        self.table.iloc[self.trap2[0],self.trap2[1]]='T'
        T = pd.DataFrame({
              'linhas':[0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3],\
              'colunas':[0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]
                          })
        self.table = self.table.replace(0,'_')
        self.table.iloc[T['linhas'][self.position],T['colunas'][self.position]] = 'o'
        return self.table
    
    def print_maze(self,table):
        print(self.table.to_string(index=False,header=False))
        print('')