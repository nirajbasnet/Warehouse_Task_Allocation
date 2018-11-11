from tkinter import Tk,Canvas
import time
import copy

class World:
    def __init__(self, master,x=5,y=5):
        self.master = master
        master.title("Warehouse Grid")
        master.bind("<Up>", self.call_up)
        master.bind("<r>", self.reset_event)
        self.x,self.y=x,y
        self.width=60
        self.board = Canvas(master, width=self.x * self.width, height=self.y * self.width)
        self.board.grid(row=0, column=0)
        self.render_grid()
        self.agents=[]
        self.agents_pos=[]
        self.agents_home_pos=[]
        self.items=[]
        self.items_pos=[]
        self.items_qty=[]
        self.bins=[]
        self.bins_pos=[]
        self.bins_qty=[]

    def call_up(self,event):
        self.step(0, "up")

    def reset_event(self,event):
        self.reset()

    def set_size(self,m,n):
        self.x,self.y=m,n
        self.board = Canvas(self.master, width=self.x * self.width, height=self.y * self.width)

    def render_grid(self):
        for i in range(self.x):
            for j in range(self.y):
                self.board.create_rectangle(i * self.width, j * self.width, (i + 1) * self.width, (j + 1) * self.width, fill="white", width=1)

    def restart_episode(self):
        print('new episode started')

    def add_agent(self,tagid,pos=[0,0],color='blue'):
        agent = self.board.create_rectangle(pos[0] * self.width + self.width * 2 / 10,
                                           pos[1] * self.width + self.width * 2 / 10,
                                           pos[0] * self.width + self.width * 8 / 10,
                                           pos[1] * self.width + self.width * 8 / 10, fill=color, width=1, tag=tagid)
        self.agents.append(agent)
        self.agents_pos.append(pos)
        self.agents_home_pos.append(list(pos))

    def add_items(self,tagid,pos=(0,0),qty=1,color='green'):
        item_unit = self.board.create_rectangle(pos[0] * self.width + self.width * 2 / 10,
                                           pos[1] * self.width + self.width * 2 / 10,
                                           pos[0] * self.width + self.width * 8 / 10,
                                           pos[1] * self.width + self.width * 8 / 10, fill=color, width=1, tag=tagid)
        qty_label = self.board.create_text((pos[0]*self.width+self.width/2,pos[1]*self.width+self.width/2),font=("Gothic", 16), text=str(qty))
        self.items.append(item_unit)
        self.items_pos.append(pos)
        self.items_qty.append(qty)

    def add_bins(self,tagid,pos=(0,0),qty=0,color='yellow'):
        bin_unit = self.board.create_rectangle(pos[0] * self.width + self.width * 2 / 10,
                                           pos[1] * self.width + self.width * 2 / 10,
                                           pos[0] * self.width + self.width * 8 / 10,
                                           pos[1] * self.width + self.width * 8 / 10, fill=color, width=1, tag=tagid)
        qty_label = self.board.create_text((pos[0]*self.width+self.width/2,pos[1]*self.width+self.width/2),font=("Gothic", 16), text=str(qty))
        self.bins.append(bin_unit)
        self.bins_pos.append(pos)
        self.bins_qty.append(qty)


    def initialize_world(self):
        self.add_items("itemA", (2, 2), 4)
        self.add_items("itemB", (7, 2), 7)
        self.add_items("itemC", (2, 7), 4)
        self.add_items("itemD", (7, 7), 7)

        self.add_bins("binA", (0, 0))
        self.add_bins("binB", (self.x-1, 0))
        self.add_bins("binC", (0, self.y-1))
        self.add_bins("binD", (self.x-1, self.y-1))

        self.add_agent("agent0",[4,4],'blue')
        self.add_agent("agent1",[5,4])
        self.add_agent("agent1",[4,5])
        self.add_agent("agent1",[5,5])

        print("agent pos: ",self.agents_pos)
        print("items pos: ", self.items_pos,self.items_qty)
        print("bins pos: ", self.bins_pos,self.bins_qty)

    def reset(self):
        self.bins_qty=[0 for i in range(len(self.bins))]
        self.items_qty=[10 for i in range(len(self.bins))]
        self.agents_pos=copy.deepcopy(self.agents_home_pos)

        for i in range(len(self.agents)):
            self.board.coords(self.agents[i], self.agents_pos[i][0] * self.width + self.width * 2 / 10,
                              self.agents_pos[i][1] * self.width + self.width * 2 / 10,
                              self.agents_pos[i][0] * self.width + self.width * 8 / 10,
                              self.agents_pos[i][1] * self.width + self.width * 8 / 10)

    def step(self,agent_id,action_cmd):
        dx,dy=self.map_action_commands(action_cmd)
        # print(self.agents_pos[agent_id][0])
        self.agents_pos[agent_id][0] += dx
        self.agents_pos[agent_id][1] += dy
        pos=self.agents_pos[agent_id]
        # print(self.agents_pos,self.agents_home_pos)
        self.board.coords(self.agents[agent_id],pos[0] * self.width + self.width * 2 / 10,
                                               pos[1] * self.width + self.width * 2 / 10,
                                               pos[0] * self.width + self.width * 8 / 10,
                                               pos[1] * self.width + self.width * 8 / 10)

    def map_action_commands(self,action_cmd):
        if action_cmd == "up":
            return 0, -1
        elif action_cmd == "down":
            return 0, 1
        elif action_cmd == "right":
            return 1, 0
        elif action_cmd == "left":
            return -1, 0
        else:
            return 0, 0


root = Tk()
def start_world():
    root.mainloop()
