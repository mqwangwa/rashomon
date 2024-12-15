import pandas as pd
import numpy as np

# Metrics calculated over Rashomon Allocations 
# for a fixed test size, selection rate, and qualification rate
# computed over each seed / iteration 

class Metrics:

	# df = allocations dataframe
	# data = original dataset of features
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

	# Number of Rashomon allocations found
	def count_rashomon_allocations(self):
		grouped = self.df.groupby(["seed", "iteration"]).count()["allocation_idx"].reset_index()
		return np.mean(grouped["allocation_idx"]), np.std(grouped["allocation_idx"])

	# Number of Rashomon models found
	def count_rashomon_models(self):
		grouped = self.df.groupby(["seed", "iteration"]).sum()["model_count"].reset_index()
		return np.mean(grouped["model_count"]), np.std(grouped["model_count"])

	# Number of Rashomon models per Rashomon allocation
	def count_rashomon_models_per_allocation(self):
		grouped = self.df.groupby(["seed", "iteration"])["model_count"].mean().reset_index()
		return np.mean(grouped["model_count"]), np.std(grouped["model_count"])

	# Homogenization -- Systemic Rejection: Number of people never selected across Rashomon allocations
	def systemic_exclusion(self):
		metric = []
		for seed in self.df["seed"].unique():
			for iteration in self.df["iteration"].unique():
				# Calculate intersection of unselected people
				allocations = self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration), "unselected"].to_list()
				systemic_rejection = set(allocations[0])
				for a in allocations:
					systemic_rejection = systemic_rejection.intersection(set(a))
				metric.append(len(systemic_rejection))

		return np.mean(metric), np.std(metric)

	# Homogenization -- Pairwise Systemic Rejection: Number of people never selected across a pair of Rashomon allocations
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
				if not np.all(np.isnan(inner)):
					metric.append(np.nanmean(metric_inner))

		return np.nanmean(metric), np.nanstd(metric)


	# Helper function for entropy-based homogenization measures
	def entropy(self, probs):
		e = 0
		for p in probs:
			if p > 0 and p < 1:
				e += p * np.log(p)
		return -e

	# Homogenization -- Entropy in feature over selected individuals
	# Baseline 1: Entropy in feature over all qualified individuals
	# Baseline 2: Weighted average of entropy over qualified and unqualified individuals, based on precision
	def homogenization_in_selected_individuals(self, feature):
		metric = []
		baseline1 = []
		baseline2 = []
		for seed in self.df["seed"].unique():
			for iteration in self.df["iteration"].unique():
				selected = self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration)&(self.df["allocation_idx"]==0), "selected"].values[0]
				unselected = self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration)&(self.df["allocation_idx"]==0), "unselected"].values[0]
				people = selected + unselected
				test_data = self.data.loc[self.data["person_id"].isin(people), ["person_id", self.QUALIFICATION_COLUMN, feature]].copy()
				
				k_prime = self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration), "k'"].mean()
				k = len(selected)

				qualified_people = test_data.loc[test_data[self.QUALIFICATION_COLUMN]==1]
				qualified_entropy = self.entropy(list(qualified_people[feature].value_counts()/len(qualified_people)))
				baseline1.append(qualified_entropy)

				unqualified_people = test_data.loc[test_data[self.QUALIFICATION_COLUMN]==0]
				unqualified_entropy = self.entropy(list(unqualified_people[feature].value_counts()/len(unqualified_people)))
				baseline2.append((qualified_entropy * (k_prime / k)) + (unqualified_entropy * ((k-k_prime)/k)))

				allocations = self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration), "selected"].to_list()
				metric_inner = []
				for a in allocations:
					selected_people = test_data.loc[test_data["person_id"].isin(a)].copy()
					metric_inner.append(self.entropy(list(selected_people[feature].value_counts()/len(selected_people))))
				metric.append(np.mean(metric_inner))
		return np.mean(metric), np.std(metric), np.mean(baseline1), np.std(baseline1), np.mean(baseline2), np.std(baseline2)

	# Homogenization -- Entropy in individual decisions over Rashomon allocations
	# Baseline: Entropy using expected selection rate, based on qualification and precision
	def homogenization_in_individual_decisions(self):
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


	# Arbitrariness: Variance in selection rate across Rashomon allocations, conditional on qualification
	def arbitrariness(self):
		qualified_std = []
		unqualified_std = []

		for seed in self.df["seed"].unique():
			for iteration in self.df["iteration"].unique():
				selected = self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration)&(self.df["allocation_idx"]==0), "selected"].values[0]
				unselected = self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration)&(self.df["allocation_idx"]==0), "unselected"].values[0]
				people = selected + unselected
				test_data = self.data.loc[self.data["person_id"].isin(people), ["person_id", self.QUALIFICATION_COLUMN]].copy()

				# Track how many times each person was selected across Rashomon allocations
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

				qualified_std.append(np.std(qualified_selections))
				unqualified_std.append(np.std(unqualified_selections))

		return np.mean(qualified_std), np.std(qualified_std), np.mean(unqualified_std), np.std(unqualified_std)

	# Group Fairness -- selection rate for demographic group over Rashomon Allocations
	# Compute average selection rate and best found selection rate
	def group_selection_rate(self, group_col, group_val):
		metric = []
		group_people = self.data.loc[self.data[group_col]==group_val, "person_id"].to_list()

		for seed in self.df["seed"].unique():
			for iteration in self.df["iteration"].unique():
				allocations = self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration), "selected"].to_list()
				metric_inner = []
				for a in allocations:
					group_selected = 0
					for p in a:
						if p in group_people:
							group_selected += 1
					metric_inner.append(group_selected/len(a))
				metric.append(metric_inner)

		metric_avg = [np.nanmean(inner) for inner in metric]
		metric_best_found = [np.nanmax(inner) for inner in metric]
		return np.nanmean(metric_avg), np.nanstd(metric_avg), np.nanmean(metric_best_found), np.nanstd(metric_best_found)

	# Group Fairness -- selection rate for demographic group over Rashomon Allocations
	# Compute best possible selection rate under fixed precision, and any better precision
	def group_selection_rate_best(self, group_col, group_val):
		metric_precision_fixed = []
		metric_precision_greater = []

		for seed in self.df["seed"].unique():
			for iteration in self.df["iteration"].unique():
				selected = self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration)&(self.df["allocation_idx"]==0), "selected"].values[0]
				unselected = self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration)&(self.df["allocation_idx"]==0), "unselected"].values[0]
				k = len(selected)
				people = selected + unselected

				test_data = self.data.loc[self.data["person_id"].isin(people)].copy()
				test_data = test_data[["person_id", self.QUALIFICATION_COLUMN, group_col]].reset_index(drop=True).copy()

				k_prime = self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration), "k'"].max()
				qualified_group = test_data[(test_data[group_col]==group_val)&(test_data[self.QUALIFICATION_COLUMN]==1)]
				unqualified_group = test_data[(test_data[group_col]==group_val)&(test_data[self.QUALIFICATION_COLUMN]==0)]

				# Select as many people from the group as posisble under the k' and (k-k') restrictions
				group_selected = 0
				if len(qualified_group) >= k_prime:
					group_selected += k_prime
				else:
					group_selected += len(qualified_group)
				if len(unqualified_group)>=(k-k_prime):
					group_selected += (k-k_prime)
				else:
					group_selected += len(unqualified_group)
				metric_precision_fixed.append(group_selected/k)

				# Select as many people from the group as possible for any k' better than the k'
				metric_inner = []
				for precision in range(k_prime, k+1):
					group_selected = 0
					if len(qualified_group) >= precision:
						group_selected += precision
					else:
						group_selected += len(qualified_group)
					if len(unqualified_group)>=(k-precision):
						group_selected += (k-precision)
					else:
						group_selected += len(unqualified_group)
					metric_inner.append(group_selected/k)
				metric_precision_greater.append(np.nanmax(metric_inner))

		return np.mean(metric_precision_fixed), np.std(metric_precision_fixed), np.mean(metric_precision_greater), np.std(metric_precision_greater)


	# Group Fairness -- Among selected, check ratio of a feature by group
	# Assumes denominator group will have smaller value of the feature
	def group_feature_ratio(self, group_col, group_val_num, group_val_denom, feature_col):
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

		metric_avg = [np.nanmean(inner) if not np.all(np.isnan(inner)) else np.nan for inner in metric]
		metric_best_found = [np.nanmin(inner) if not np.all(np.isnan(inner)) else np.nan for inner in metric]
		return np.nanmean(metric_avg), np.nanstd(metric_avg), np.nanmean(metric_best_found), np.nanstd(metric_best_found)


	def alternate_within_ties(self, group, alt_col):
		return group.sort_values(by=alt_col, key=lambda x: x.rank(method='first') % 2)

	# Group Fairness -- Among selected, what is the best possible ratio of a feature by group
	def group_feature_ratio_best(self, group_col, group_val_num, group_val_denom, feature_col):
		metric_precision_fixed = []
		metric_precision_greater = []
		for seed in self.df["seed"].unique():
			for iteration in self.df["iteration"].unique():
				selected = self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration)&(self.df["allocation_idx"]==0), "selected"].values[0]
				unselected = self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration)&(self.df["allocation_idx"]==0), "unselected"].values[0]
				people = selected + unselected

				k = len(selected)
				k_prime = self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration), "k'"].max()

				test_data = self.data.loc[self.data["person_id"].isin(people)].copy()
				test_data = test_data[["person_id", self.QUALIFICATION_COLUMN, group_col, feature_col]].reset_index(drop=True).copy()
				
				qualified = test_data[(test_data[self.QUALIFICATION_COLUMN]==1)].copy()
				unqualified = test_data[(test_data[self.QUALIFICATION_COLUMN]==0)].copy()

				qualified = qualified.sort_values(by=[feature_col], ascending=False).reset_index(drop=True)
				unqualified = unqualified.sort_values(by=[feature_col], ascending=False).reset_index(drop=True)

				# Select based on highest feature value, under k' and (k-k') restrictions
				metric_inner = []
				for precision in range(k_prime, k+1):
					if len(qualified) < precision:
						continue
					min_feature = qualified.loc[:precision-1, feature_col].min()
					qualified_in = qualified.loc[qualified[feature_col]>min_feature].copy()
					qualified_ties = qualified.loc[qualified[feature_col]==min_feature].copy()

					min_feature = unqualified.loc[:max((k-precision-1),0), feature_col].min()
					unqualified_in = unqualified.loc[unqualified[feature_col]>min_feature].copy()
					unqualified_ties = unqualified.loc[unqualified[feature_col]==min_feature].copy()

					selected = []
					if len(qualified_in)==0:
						selected.append(qualified_ties.sample(n=precision))
					elif len(qualified_in) <= precision:
						selected.append(qualified_in)
						selected.append(qualified_ties.sample(n=precision-len(qualified_in)))


					if precision < k:
						if len(unqualified_in)==0:
							selected.append(unqualified_ties.sample(n=k-precision))
						elif len(unqualified_in) <= k-precision:
							selected.append(unqualified_in)
							selected.append(unqualified_ties.sample(n=k-precision-len(unqualified_in)))

					selected = pd.concat(selected)

					feature_group_num = selected.loc[(selected[group_col]==0), feature_col].mean()
					feature_group_denom = selected.loc[(selected[group_col]==1), feature_col].mean()
					metric_inner.append(feature_group_num/feature_group_denom)

				metric_precision_fixed.append(metric_inner[0])
				if np.sum(np.isnan(metric_inner)) != len(metric_inner):
					metric_precision_greater.append(np.nanmin(metric_inner))
				'''
				qualified_g1 = test_data[(test_data[self.QUALIFICATION_COLUMN]==1)&(test_data[group_col]==group_val_num)].copy().sort_values(by=[feature_col], ascending=False).reset_index(drop=True)
				qualified_g2 = test_data[(test_data[self.QUALIFICATION_COLUMN]==1)&(test_data[group_col]==group_val_denom)].copy().sort_values(by=[feature_col], ascending=False).reset_index(drop=True)
				unqualified_g1 = test_data[(test_data[self.QUALIFICATION_COLUMN]==0)&(test_data[group_col]==group_val_num)].copy().sort_values(by=[feature_col], ascending=False).reset_index(drop=True)
				unqualified_g2 = test_data[(test_data[self.QUALIFICATION_COLUMN]==0)&(test_data[group_col]==group_val_denom)].copy().sort_values(by=[feature_col], ascending=False).reset_index(drop=True)

				qualified = test_data[(test_data[self.QUALIFICATION_COLUMN]==1)].copy().sort_values(by=[feature_col], ascending=False).reset_index(drop=True)
				unqualified = test_data[(test_data[self.QUALIFICATION_COLUMN]==0)].copy().sort_values(by=[feature_col], ascending=False).reset_index(drop=True)

				qualified = qualified.groupby(feature_col, group_keys=False).apply(self.alternate_within_ties, alt_col=group_col)
				unqualified = unqualified.groupby(feature_col, group_keys=False).apply(self.alternate_within_ties, alt_col=group_col)

				selected = pd.concat([
					qualified.loc[:k_prime-1],
					unqualified.loc[:max((k-k_prime)-1, 0)]
				])

				#if len(qualified_g1)==0 or len(qualified_g2)==0 or len(unqualified_g1)==0 or len(unqualified_g2)==0:
				#	continue

				q_g1_g2 = len(qualified_g1)/(len(qualified_g1)+len(qualified_g2))
				uq_g1_g2 = len(unqualified_g1)/(len(unqualified_g1)+len(unqualified_g2))

				q_g1 = round(k_prime*q_g1_g2)
				q_g2 = k_prime - q_g1
				uq_g1 = round((k-k_prime)*uq_g1_g2)
				uq_g2 = k - k_prime - uq_g1

				#selected = pd.concat([
				#	qualified_g1.loc[:q_g1-1],
				#	qualified_g2.loc[:q_g2-1],
				#	unqualified_g1.loc[:uq_g1-1],
				#	unqualified_g2.loc[:uq_g2-1]
				#])	

				feature_group_num = selected.loc[(selected[group_col]==group_val_num), feature_col].mean()
				feature_group_denom = selected.loc[(selected[group_col]==group_val_denom), feature_col].mean()
				metric_precision_fixed.append(feature_group_num/feature_group_denom)

				#metric_inner = []
				#for precision in range(k_prime, k+1):
					#q_g1 = round(precision*q_g1_g2)
					#q_g2 = precision - q_g1
					#uq_g1 = round((k-precision)*uq_g1_g2)
					#uq_g2 = k - precision - uq_g1

					#selected = pd.concat([
					#	qualified_g1.loc[:q_g1-1],
					#	qualified_g2.loc[:q_g2-1],
					#	unqualified_g1.loc[:uq_g1-1],
					#	unqualified_g2.loc[:uq_g2-1]
					#])	

					#feature_group_num = selected.loc[(selected[group_col]==group_val_num), feature_col].mean()
					#feature_group_denom = selected.loc[(selected[group_col]==group_val_denom), feature_col].mean()
					#metric_inner.append(feature_group_num/feature_group_denom)
				#metric_precision_greater.append(np.nanmin(metric_inner))
				'''

		return np.nanmean(metric_precision_fixed), np.nanstd(metric_precision_fixed), np.nanmean(metric_precision_greater), np.nanstd(metric_precision_greater)


