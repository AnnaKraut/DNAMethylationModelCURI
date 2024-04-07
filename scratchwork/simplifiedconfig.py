def birth_event(self):
        self.narr[self.currstep] = self.narr[self.currstep-1] + 1
        print("ran birth, updated self.n to", self.narr[self.currstep])
def death_event(self):
    self.narr[self.currstep] = self.narr[self.currstep-1] - 1
    print("ran deatj, updated self.n to", self.narr[self.currstep])

base_rates = {"birth":0.1,
                  "death":0.05
}
    
base_events = {"birth":birth_event,
                   "death":death_event
}

def birth_rate(self):
      return self.narr[self.currstep - 1]*base_rates["birth"]

def death_rate(self):
      return self.narr[self.currstep - 1]*base_rates["death"]

rate_calculation = {"birth":birth_rate,
                    "death":death_rate
}