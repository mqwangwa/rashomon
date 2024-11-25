import pandas as pd
import numpy as np

# Note: All metrics calculated separately for each seed/iteration
class Metrics:

	def __init__(self, df, data, QUALIFICATION_COLUMN):
		self.df = df
		self.data = data
		self.QUALIFICATION_COLUMN = QUALIFICATION_COLUMN


	# Number of qualified people selected (k')
	def k_prime(self):
		metric = []
		for seed in self.df["seed"].unique():
			for iteration in self.df["iteration"].unique():
				metric.append(self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration), "k'"].mean())

		return np.mean(metric), np.std(metric)

	# Number of qualified people in test set (n')
	def n_prime(self):
		metric = []
		for seed in self.df["seed"].unique():
			for iteration in self.df["iteration"].unique():
				metric.append(self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration), "n'"].mean())

		return np.mean(metric), np.std(metric)

	def count_rashomon_allocations(self):
		grouped = self.df.groupby(["seed", "iteration"]).count()["allocation_idx"].reset_index()
		return np.mean(grouped["allocation_idx"]), np.std(grouped["allocation_idx"])

	def count_rashomon_models(self):
		grouped = self.df.groupby(["seed", "iteration"]).sum()["model_count"].reset_index()
		return np.mean(grouped["model_count"]), np.std(grouped["model_count"])

	def count_rashomon_models_per_allocation(self):
		grouped = self.df.groupby(["seed", "iteration"])["model_count"].mean().reset_index()
		return np.mean(grouped["model_count"]), np.std(grouped["model_count"])

	# Homogenization -- Systemic Exclusion (Number of people never selected)
	def systemic_exclusion(self):
		metric = []
		for seed in self.df["seed"].unique():
			for iteration in self.df["iteration"].unique():
				# Calculate intersection of rejected people across Rashomon allocations
				allocations = self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration), "unselected"].to_list()
				systemic_rejection = set(allocations[0])
				for a in allocations:
					systemic_rejection = systemic_rejection.intersection(set(a))
				metric.append(len(systemic_rejection))

		return np.mean(metric), np.std(metric)

	# Homogenization -- Systemic Exclusion (Number of people never selected)
	def systemic_exclusion_pairwise(self):
		metric = []
		for seed in self.df["seed"].unique():
			for iteration in self.df["iteration"].unique():
				allocations = self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration), "unselected"].to_list()
				metric_inner = []
				for a in allocations:
					for a2 in allocations:
						if a == a2:
							continue
						systemic_rejection = set(a).intersection(set(a2))
						metric_inner.append(len(systemic_rejection))
				metric.append(np.nanmean(metric_inner))

		return np.nanmean(metric), np.nanstd(metric)


	def entropy(self, probs):
		e = 0
		for p in probs:
			if p > 0 and p < 1:
				e += p * np.log(p)
		return -e


	def local_homogenization(self):
		metric = []
		baseline = []
		for seed in self.df["seed"].unique():
			for iteration in self.df["iteration"].unique():
				selected = self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration)&(self.df["allocation_idx"]==0), "selected"].values[0]
				unselected = self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration)&(self.df["allocation_idx"]==0), "unselected"].values[0]
				people = selected + unselected
				selected_counts = dict.fromkeys(people, 0)

				n_prime = self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration), "n'"].mean()
				k_prime = self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration), "k'"].mean()
				n = len(people)
				k = len(selected)
				qualified_prob = k_prime / n_prime
				qualified_entropy = self.entropy([qualified_prob, 1 - qualified_prob])
				unqualified_prob = (k - k_prime) / (n - n_prime)
				unqualified_entropy = self.entropy([unqualified_prob, 1-unqualified_prob])
				baseline.append((qualified_entropy * (n_prime / n)) + (unqualified_entropy * ((n-n_prime)/n)))
				
				allocations = self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration), "selected"].to_list()
				for a in allocations:
					for p in a:
						selected_counts[p] += 1

				selected_counts = {key: self.entropy([value / len(allocations), 1 - (value / len(allocations))]) for key, value in selected_counts.items()}
				metric.append(np.mean(list(selected_counts.values())))

		return np.mean(metric), np.std(metric), np.mean(baseline), np.std(baseline)


	def global_homogenization(self, feature):
		metric = []
		baseline = []
		for seed in self.df["seed"].unique():
			for iteration in self.df["iteration"].unique():
				selected = self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration)&(self.df["allocation_idx"]==0), "selected"].values[0]
				unselected = self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration)&(self.df["allocation_idx"]==0), "unselected"].values[0]
				people = selected + unselected
				test_data = self.data.loc[self.data["person_id"].isin(people)].copy()
				
				n = len(people)
				n_prime = self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration), "n'"].mean()

				qualified_people = test_data.loc[test_data[self.QUALIFICATION_COLUMN]==1]
				qualified_entropy = self.entropy(list(qualified_people[feature].value_counts()/len(qualified_people)))
				unqualified_people = test_data.loc[test_data[self.QUALIFICATION_COLUMN]==0]
				unqualified_entropy = self.entropy(list(unqualified_people[feature].value_counts()/len(unqualified_people)))
				baseline.append((qualified_entropy * (n_prime / n)) + (unqualified_entropy * ((n-n_prime)/n)))


				allocations = self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration), "selected"].to_list()
				metric_inner = []
				for a in allocations:
					selected_data = test_data.loc[test_data["person_id"].isin(a)].copy()
					type_probs = list(selected_data[feature].value_counts()/len(selected_data))
					metric_inner.append(self.entropy(type_probs))
				metric.append(np.mean(metric_inner))
		return np.mean(metric), np.std(metric), np.mean(baseline), np.std(baseline)



	# Individual Fairness -- Number of selections across Rashomon allocations, conditional on qualification
	def selections_by_qualification(self):
		qualified_avg = []
		qualified_std = []
		unqualified_avg = []
		unqualified_std = []

		for seed in self.df["seed"].unique():
			for iteration in self.df["iteration"].unique():
				selected = self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration)&(self.df["allocation_idx"]==0), "selected"].values[0]
				unselected = self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration)&(self.df["allocation_idx"]==0), "unselected"].values[0]
				people = selected + unselected
				test_data = self.data.loc[self.data["person_id"].isin(people)].copy()

				# Track how many times each person was selected across allocations
				qualified_selections = dict.fromkeys(test_data.loc[test_data[self.QUALIFICATION_COLUMN]==1, "person_id"].to_list(), 0)
				unqualified_selections = dict.fromkeys(test_data.loc[test_data[self.QUALIFICATION_COLUMN]==0, "person_id"].to_list(), 0)
				allocations = self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration), "selected"].to_list()
				for a in allocations:
					for p in a:
						if p in qualified_selections:
							qualified_selections[p] += 1
						else:
							unqualified_selections[p] += 1

				qualified_selections = np.array(list(qualified_selections.values()))/len(allocations)
				unqualified_selections = np.array(list(unqualified_selections.values()))/len(allocations)

				qualified_avg.append(np.mean(qualified_selections))
				qualified_std.append(np.std(qualified_selections))
				unqualified_avg.append(np.mean(unqualified_selections))
				unqualified_std.append(np.std(unqualified_selections))

		return np.mean(qualified_avg), np.mean(qualified_std), np.mean(unqualified_avg), np.mean(unqualified_std)

	# Group Fairness -- % of minority selected 
	def minority_selection_rate(self, group_col, group_val):
		metric = []
		minority_people = self.data.loc[self.data[group_col]==group_val, "person_id"].to_list()

		for seed in self.df["seed"].unique():
			for iteration in self.df["iteration"].unique():
				allocations = self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration), "selected"].to_list()
				metric_inner = []
				for a in allocations:
					minority_selected = 0
					for p in a:
						if p in minority_people:
							minority_selected += 1
					metric_inner.append(minority_selected/len(a))
				metric.append(metric_inner)

		metric_flattened = [np.nanmean(inner) for inner in metric]
		metric_best_empirical = [max(inner) for inner in metric]
		return np.nanmean(metric_flattened), np.nanstd(metric_flattened), np.nanmean(metric_best_empirical)

	# Group Fairness -- % of minority selected (best possible)
	def best_minority_selection_rate(self, group_col, group_val):
		metric = []
		for seed in self.df["seed"].unique():
			for iteration in self.df["iteration"].unique():
				selected = self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration)&(self.df["allocation_idx"]==0), "selected"].values[0]
				k = len(selected)
				unselected = self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration)&(self.df["allocation_idx"]==0), "unselected"].values[0]
				people = selected + unselected
				
				test_data = self.data.loc[self.data["person_id"].isin(people)].copy()
				test_data = test_data[["person_id", self.QUALIFICATION_COLUMN, group_col]].reset_index(drop=True).copy()

				k_prime = self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration), "k'"].max()
				qualified_minority = test_data[(test_data[group_col]==group_val)&(test_data[self.QUALIFICATION_COLUMN]==1)]
				unqualified_minority = test_data[(test_data[group_col]==group_val)&(test_data[self.QUALIFICATION_COLUMN]==0)]

        		# Select as many minority people as possible under the k' and (k-k') restrictions
				minority_selected = 0
				if len(qualified_minority)>=k_prime:
					minority_selected += k_prime
				else:
					minority_selected += len(qualified_minority)
				if len(unqualified_minority)>=(k-k_prime):
					minority_selected += (k-k_prime)
				else:
					minority_selected += len(unqualified_minority)
				metric.append(minority_selected/k)
		return np.mean(metric), np.std(metric)

	# Group Fairness -- Among selected, check ratio of a feature by group
	# Assumes denominator group will have smaller value of the feature
	def feature_ratio_by_group(self, group_col, group_val_num, group_val_denom, feature_col):
		metric = []
		for seed in self.df["seed"].unique():
			for iteration in self.df["iteration"].unique():
				selected = self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration)&(self.df["allocation_idx"]==0), "selected"].values[0]
				unselected = self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration)&(self.df["allocation_idx"]==0), "unselected"].values[0]
				people = selected + unselected
				
				test_data = self.data.loc[self.data["person_id"].isin(people)].copy()

				allocations = self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration), "selected"].to_list()
				metric_inner = []
				for a in allocations:
					feature_group_num = test_data.loc[(test_data[group_col]==group_val_num)&(test_data["person_id"].isin(a)), feature_col].mean()
					feature_group_denom = test_data.loc[(test_data[group_col]==group_val_denom)&(test_data["person_id"].isin(a)), feature_col].mean()
					metric_inner.append(feature_group_num / feature_group_denom)
				metric.append(metric_inner)

		metric_flattened = [np.nanmean(inner) for inner in metric]
		metric_best_empirical = [min(inner) for inner in metric] 
		return np.nanmean(metric_flattened), np.nanstd(metric_flattened), np.nanmean(metric_best_empirical)

	# Group Fairness -- Among selected, best ratio of a feature by group
	def best_feature_ratio_by_group(self, group_col, group_val_num, group_val_denom, feature_col):
		metric = []
		for seed in self.df["seed"].unique():
			for iteration in self.df["iteration"].unique():
				selected = self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration)&(self.df["allocation_idx"]==0), "selected"].values[0]
				k = len(selected)
				unselected = self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration)&(self.df["allocation_idx"]==0), "unselected"].values[0]
				people = selected + unselected

				test_data = self.data.loc[self.data["person_id"].isin(people)].copy()
				test_data = test_data[["person_id", self.QUALIFICATION_COLUMN, group_col, feature_col]].reset_index(drop=True).copy()

				k_prime = self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration), "k'"].max()
				qualified = test_data[(test_data[self.QUALIFICATION_COLUMN]==1)]
				unqualified = test_data[(test_data[self.QUALIFICATION_COLUMN]==0)]

				qualified = qualified.sort_values(by=[feature_col], ascending=False).reset_index(drop=True)
				unqualified = unqualified.sort_values(by=[feature_col], ascending=False).reset_index(drop=True)

				# Select based on highest feature value, under k' and (k-k') restrictions
				selected = pd.concat([qualified.loc[:k_prime-1], unqualified.loc[:(k-k_prime)-1]])
				feature_group_num = selected.loc[(test_data[group_col]==0), feature_col].mean()
				feature_group_denom = selected.loc[(test_data[group_col]==1), feature_col].mean()
				if feature_group_num == 0 and feature_group_denom == 0:
					metric.append(1)
				elif feature_group_denom == 0:
					metric.append(np.nan)
				else:
					metric.append(feature_group_num/feature_group_denom)
		return np.nanmean(metric), np.nanstd(metric)

