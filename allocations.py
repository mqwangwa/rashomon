import pandas as pd
from tqdm import tqdm
import numpy as np

class Allocations:
	def __init__(self, df, TEST_SIZE, SELECTION_RATE, QUALIFICATION_COLUMN, RASHOMON_EPSILON, ITERATIONS_PER_SPLIT):
		self.df = df
		self.TEST_SIZE = TEST_SIZE
		self.SELECTION_RATE = SELECTION_RATE
		self.RASHOMON_EPSILON = RASHOMON_EPSILON
		self.ITERATIONS_PER_SPLIT = ITERATIONS_PER_SPLIT
		self.k = int(self.SELECTION_RATE*self.TEST_SIZE)

		self.NUM_MODELS = len([c for c in self.df.columns if c.startswith("m_")])
		self.NUM_SPLITS = len(self.df["seed"].unique())

		self.QUALIFICATION_COLUMN = QUALIFICATION_COLUMN

	def calculate_rashomon_allocations(self, seed, iteration, df, perf):
		best_perf = 1
		for model in range(1, self.NUM_MODELS+1):
			model_perf = float(perf.loc[perf["person_id"]==-1, "m_"+str(model)])
			if model_perf < best_perf:
				best_perf = model_perf

		allocations = {}
		allocation_data = []
		allocation_idx = 0
		for model in range(1, self.NUM_MODELS+1):
			if float(perf.loc[perf["person_id"]==-1, "m_"+str(model)]) > best_perf + self.RASHOMON_EPSILON:
				continue

			selected_idx = df["m_"+str(model)].nlargest(self.k).index.to_list()
			k_prime = int(df.loc[selected_idx, self.QUALIFICATION_COLUMN].sum())
			n_prime = int(df[self.QUALIFICATION_COLUMN].sum())
			selected_people = df.loc[selected_idx, "person_id"].to_list()

			selected_people.sort()
			allocation = tuple(selected_people)
			if allocation not in allocations:
				allocation_data.append({
					"seed":seed,
					"iteration":iteration,
					"allocation_idx":allocation_idx,
					"selected": selected_people,
					"k'": k_prime,
					"n'": n_prime,
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
		allocations = []
		people = []
		for split in range(self.NUM_SPLITS):
			split_df = self.df[(self.df["seed"]==split)&(self.df["person_id"]>=0)].copy()
			perf_df = self.df[(self.df["seed"]==split)&(self.df["person_id"]<0)].copy()

			for i in range(self.ITERATIONS_PER_SPLIT):
				people_sample = split_df.sample(n=self.TEST_SIZE, random_state=i)
				people.append({"seed": split, "iteration": i, "people": people_sample["person_id"].to_list()})
				allocations += self.calculate_rashomon_allocations(split, i, people_sample, perf_df)
		people = pd.DataFrame(people)
		allocations = pd.DataFrame(allocations)
		return allocations, people
