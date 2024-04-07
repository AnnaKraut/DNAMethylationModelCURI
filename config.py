def maintenance_event(self): 
      self.methylated[self.currstep] = self.methylated[self.currstep-1]+1
      self.unmethylated[self.currstep] = self.unmethylated[self.currstep-1]

def denovo_event(self):
      self.methylated[self.currstep] = self.methylated[self.currstep-1]
      self.unmethylated[self.currstep] = self.unmethylated[self.currstep-1]-1

def demaintenance_event(self):
      self.methylated[self.currstep] = self.methylated[self.currstep-1]
      self.unmethylated[self.currstep] = self.unmethylated[self.currstep-1]+1

def demethylation_event(self):
      self.methylated[self.currstep] = self.methylated[self.currstep-1]-1
      self.unmethylated[self.currstep] = self.unmethylated[self.currstep-1]

def birth_event(self):
      self.methylated[self.currstep] = 0
      #calculate hemimethylated and draw a random binomial to determine how many become unmethylated
      hemimethylated = self.population - (self.methylated[self.currstep-1] + self.unmethylated[self.currstep-1])
      newly_unmethylated = self.rng.binomial(hemimethylated, 0.5)
      self.unmethylated[self.currstep] = self.unmethylated[self.currstep-1] + newly_unmethylated

base_events = {"maintenance methylation":maintenance_event,
               "de novo methylation":denovo_event,
               "maintenance demethylation":demaintenance_event,
               "demethylation":demethylation_event,
               "birth":birth_event
}

def maintenance_rate(self):
      r_nm = 0.05
      m = self.population
      hemimethylated = m - (self.methylated[self.currstep-1] + self.unmethylated[self.currstep-1])
      rate = r_nm * hemimethylated
      return rate
def denovo_rate(self):
      r_uh = 0.05
      rate =r_uh * self.unmethylated[self.currstep-1]
      return rate
def demaintenance_rate(self):
      r_hu = 0.05
      m = self.population
      hemimethylated = m - (self.methylated[self.currstep-1] + self.unmethylated[self.currstep-1])
      rate = r_hu * hemimethylated
      return rate
def demethylation_rate(self):
      r_mh = 0.05
      rate = r_mh * self.methylated[self.currstep-1]
      return rate
def birth_rate(self):
      b = 0.001
      rate = b
      return rate

rate_calculation = {"maintenance methylation":maintenance_rate,
                    "de novo methylation":denovo_rate,
                    "maintenance demethylation":demaintenance_rate,
                    "demethylation":demethylation_rate,
                    "birth":birth_rate
}