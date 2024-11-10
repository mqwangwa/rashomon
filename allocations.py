import pandas as pd
from tqdm import tqdm
import numpy as np

class Allocations:
	def __init__(self, df, TEST_SIZE, SELECTION_RATE, QUALIFICATION_RATE, RASHOMON_EPSILON, ITERATIONS_PER_SPLIT):
		self.df = df
		self.TEST_SIZE = TEST_SIZE
		self.SELECTION_RATE = SELECTION_RATE
		self.RASHOMON_EPSILON = RASHOMON_EPSILON
		self.ITERATIONS_PER_SPLIT = ITERATIONS_PER_SPLIT
		self.k = int(self.SELECTION_RATE*self.TEST_SIZE)

		self.NUM_MODELS = len([c for c in self.df.columns if "m_" in c])
		self.NUM_SPLITS = len(self.df["seed"].unique())

		if QUALIFICATION_RATE==0.50:
			self.QUALIFICATION_COLUMN = "threshold_50"
		elif QUALIFICATION_RATE==0.25:
			self.QUALIFICATION_COLUMN = "threshold_75"
		elif QUALIFICATION_RATE==0.75:
			self.QUALIFICATION_COLUMN = "threshold_25"

	def calculate_rashomon_allocations(self, seed, iteration, df):
		k_prime = {}
		for model in range(1, self.NUM_MODELS+1):
			allocation = df["m_"+str(model)].nlargest(self.k).index.to_list()
			k_prime[model] = int(df.loc[allocation, self.QUALIFICATION_COLUMN].sum())
		best_k_prime = max(k_prime.values())

		allocations = {}
		allocation_data = []
		allocation_idx = 0
		for model in range(1, self.NUM_MODELS+1):
			if k_prime[model]/self.k < (best_k_prime/self.k)-self.RASHOMON_EPSILON:
				continue

			selected = df.loc[df["m_"+str(model)].nlargest(self.k).index, "idx"].tolist()
			selected.sort()
			allocation = tuple(selected)
			if allocation not in allocations:
				unselected = [i for i in df["idx"] if i not in selected] 
				allocation_data.append({
					"seed":seed,
					"iteration":iteration,
					"allocation_idx":allocation_idx,
					"selected": selected,
					"unselected": unselected,
					"k'": k_prime[model],
					"n'": df[self.QUALIFICATION_COLUMN].sum()
				})
				allocations[allocation] = {"allocation_idx": allocation_idx, "model_count": 1}
				allocation_idx += 1
			else:
				allocations[allocation]["model_count"] += 1

		model_counts = {}
		for v in allocations.values():
			model_counts[v["allocation_idx"]] = v["model_count"]

		for i in range(len(allocation_data)):
			allocation_data[i]["model_count"] = model_counts[allocation_data[i]["allocation_idx"]] 

		return allocation_data		

	def get_allocations(self):
		allocation_data = []
		for split in range(self.NUM_SPLITS):
			split_df = self.df[self.df["seed"]==split].copy()

			for i in range(self.ITERATIONS_PER_SPLIT):
				allocation_data += self.calculate_rashomon_allocations(split, i, split_df.sample(n=self.TEST_SIZE, random_state=i))
		allocation_data = pd.DataFrame(allocation_data)
		return allocation_data
