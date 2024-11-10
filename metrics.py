import pandas as pd
import numpy as np

# Note: All metrics calculated separately for each seed/iteration
class Metrics:

	def __init__(self, df, data, QUALIFICATION_RATE):
		self.df = df
		self.data = data
		if QUALIFICATION_RATE==0.50:
			self.QUALIFICATION_COLUMN = "threshold_50"
		elif QUALIFICATION_RATE==0.25:
			self.QUALIFICATION_COLUMN = "threshold_75"
		elif QUALIFICATION_RATE==0.75:
			self.QUALIFICATION_COLUMN = "threshold_25"


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

				qualified_avg.append(np.mean(list(qualified_selections.values())))
				qualified_std.append(np.std(list(qualified_selections.values())))
				unqualified_avg.append(np.mean(list(unqualified_selections.values())))
				unqualified_std.append(np.std(list(unqualified_selections.values())))		

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
				metric.append(feature_group_num/feature_group_denom)
		return np.nanmean(metric), np.nanstd(metric)

