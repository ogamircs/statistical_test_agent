
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.stats.proportion as smprop
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

class ABAgent:
    def __init__(self):
        self.data = None
        self.original_data = None
        self.col_mapping = {}
        self.report = []
        self.is_balanced = True
        self.rebalancing_model = None

    def log(self, message):
        """Adds a message to the agent's internal log."""
        self.report.append(message)
        print(f"[Agent]: {message}")

    def load_data(self, file_buffer):
        """Loads data from a file-like object (uploaded file)."""
        try:
            self.data = pd.read_csv(file_buffer)
            self.original_data = self.data.copy()
            self.log("Data loaded successfully.")
            return True, f"Loaded {len(self.data)} rows."
        except Exception as e:
            return False, str(e)

    def guess_columns(self):
        """Heuristically determines column roles."""
        cols = self.data.columns.str.lower()
        mapping = {}
        
        # ID detection
        id_candidates = [c for c in cols if any(x in c for x in ['id', 'user', 'cust'])]
        mapping['id'] = self.data.columns[list(cols).index(id_candidates[0])] if id_candidates else self.data.columns[0]
        
        # Group/Variant detection
        group_candidates = [c for c in cols if any(x in c for x in ['group', 'variant', 'exper'])]
        if group_candidates:
             mapping['group'] = self.data.columns[list(cols).index(group_candidates[0])]
        else:
            # Look for columns with exactly 2 or 3 unique string values, likely 'control' vs 'test'
            for c in self.data.columns:
                if self.data[c].dtype == 'object' and self.data[c].nunique() in [2, 3]:
                     mapping['group'] = c
                     break
        
        # Metric/Target detection (usually continuous or binary)
        # We look for 'revenue', 'convert', 'effect', 'buy'
        metric_candidates = [c for c in cols if any(x in c for x in ['rev', 'convert', 'is_', 'effect', 'amount'])]
        
        # Prefer columns that don't start with 'pre' or 'old'
        better_candidates = [c for c in metric_candidates if not c.startswith('pre') and 'old' not in c]
        
        if better_candidates:
            mapping['metric'] = self.data.columns[list(cols).index(better_candidates[0])]
        elif metric_candidates:
            mapping['metric'] = self.data.columns[list(cols).index(metric_candidates[0])]
        else:
             # Default to the last numeric column that isn't ID or group
             numeric_cols = self.data.select_dtypes(include=[np.number]).columns
             reserved = [mapping.get('id'), mapping.get('group')]
             remain = [c for c in numeric_cols if c not in reserved]
             if remain:
                 mapping['metric'] = remain[-1]

        # Pre-Experiment Metric detection (for DiD)
        # Look for 'pre' columns
        pre_candidates = [c for c in cols if 'pre' in c]
        if pre_candidates:
             # Try to match with metric name if possible
             metric_name = mapping.get('metric', '').lower()
             best_pre = None
             for pre in pre_candidates:
                 # Clean "pre_" prefix to see if it matches metric
                 clean_pre = pre.replace('pre_', '').replace('pre', '')
                 if clean_pre in metric_name and clean_pre != '':
                     best_pre = self.data.columns[list(cols).index(pre)]
                     break
             
             if best_pre:
                 mapping['pre_metric'] = best_pre
             else:
                 # Fallback to first 'pre' column
                 mapping['pre_metric'] = self.data.columns[list(cols).index(pre_candidates[0])]

        # Segment/Covariate detection
        # Exclude ID, Group, Metric, Pre-Metric and any other columns that look like metrics to avoid data leakage
        reserved = [v for k, v in mapping.items()]
        potential_outcomes = [c for c in self.data.columns if any(x in c.lower() for x in ['post', 'effect', 'rev', 'conv']) and c not in reserved]
        
        features = [c for c in self.data.columns if c not in reserved and c not in potential_outcomes]
        mapping['features'] = features

        # Primary Segment detection (for stratified analysis)
        # ... (rest of segment logic) ...
        # Priority: Exact 'customer_segment' or similar specific names
        priority_segments = [c for c in features if c.lower() in ['customer_segment', 'segment', 'customer_group']]
        
        if priority_segments:
             mapping['segment'] = priority_segments[0]
        else:
             segment_candidates = [c for c in features if any(x in c.lower() for x in ['seg', 'tier', 'clus', 'cat'])]
             if segment_candidates:
                  mapping['segment'] = segment_candidates[0]
             elif features:
                  # Default to first categorical feature if exists, else first feature
                  cat_feats = [c for c in features if self.data[c].dtype == 'object']
                  if cat_feats: mapping['segment'] = cat_feats[0]
                  else: mapping['segment'] = features[0]
        
        self.col_mapping = mapping
        self.log(f"Guessed columns: {mapping}")
        return mapping

    def check_pre_balance(self):
        """
        Runs an A/A test on the pre-experiment metric.
        If imbalanced, sets flag to use Diff-in-Diff.
        """
        self.use_did = False
        pre_col = self.col_mapping.get('pre_metric')
        group_col = self.col_mapping.get('group')
        
        if not pre_col or not group_col:
            self.log("Skipping Pre-Experiment Balance check (A/A test): Missing columns.")
            return

        groups = self.data[group_col].unique()
        lower_groups = [g.lower() for g in groups]
        if 'control' in lower_groups: control_label = groups[lower_groups.index('control')]
        elif 'ctrl' in lower_groups: control_label = groups[lower_groups.index('ctrl')]
        else: control_label = groups[0]
        
        control_data = self.data[self.data[group_col] == control_label][pre_col]
        
        for g in groups:
            if g == control_label: continue
            
            treat_data = self.data[self.data[group_col] == g][pre_col]
            
            # T-test for Equality of Means on Pre-Experiment Data
            t_stat, p_val = stats.ttest_ind(control_data, treat_data, equal_var=False)
            
            if p_val < 0.05:
                self.log(f"⚠️ A/A Test Failed: Pre-experiment imbalance between {control_label} and {g} on '{pre_col}' (p={p_val:.4f}).")
                self.log("👉 Switching to Difference-in-Differences (DiD) for Frequentist analysis.")
                self.use_did = True
                return
            
        self.log(f"✅ A/A Test Passed: Pre-experiment data '{pre_col}' is balanced. Using standard T-tests.")

    def check_balance(self):
        """Checks if covariates are balanced across groups."""
        group_col = self.col_mapping.get('group')
        features = self.col_mapping.get('features')
        
        if not group_col or not features:
            self.log("Cannot check balance: Missing group or feature columns.")
            self.is_balanced = True
            return

        groups = self.data[group_col].unique()
        if len(groups) < 2:
            self.log("Not enough groups for balance check.")
            return

        imbalanced_features = []
        
        # Determine Control Group
        lower_groups = [g.lower() for g in groups]
        if 'control' in lower_groups:
             control_idx = lower_groups.index('control')
             control_label = groups[control_idx]
        elif 'ctrl' in lower_groups:
             control_idx = lower_groups.index('ctrl')
             control_label = groups[control_idx]
        elif 'base' in lower_groups:
             control_idx = lower_groups.index('base')
             control_label = groups[control_idx]
        else:
             control_label = groups[0]
        
        self.log(f"Assuming '{control_label}' is Control group for balance checks.")
        
        control_df = self.data[self.data[group_col] == control_label]
        
        for feat in features:
            if feat not in self.data.columns: continue
            is_numeric = pd.api.types.is_numeric_dtype(self.data[feat])
            
            for g in groups:
                if g == control_label: continue
                
                treatment_df = self.data[self.data[group_col] == g]
                
                if is_numeric:
                    # Standardized Mean Difference (SMD)
                    mean_c = control_df[feat].mean()
                    mean_t = treatment_df[feat].mean()
                    std_pool = np.sqrt((control_df[feat].var() + treatment_df[feat].var()) / 2)
                    
                    if std_pool == 0: smd = 0
                    else: smd = abs(mean_t - mean_c) / std_pool
                    
                    if smd > 0.1: # Threshold for imbalance
                        self.log(f"⚠️ Imbalance detected in '{feat}' between {control_label} and {g} (SMD: {smd:.3f})")
                        imbalanced_features.append(feat)
                else:
                    # Chi-square for categorical
                    if self.data[feat].nunique() > 50: continue # Skip high cardinality

                    contingency = pd.crosstab(self.data[self.data[group_col].isin([control_label, g])][feat], 
                                              self.data[self.data[group_col].isin([control_label, g])][group_col])
                    chi2, p, _, _ = stats.chi2_contingency(contingency)
                    if p < 0.05:
                        self.log(f"⚠️ Imbalance detected in '{feat}' between {control_label} and {g} (p={p:.3f})")
                        imbalanced_features.append(feat)

        if imbalanced_features:
            self.is_balanced = False
            self.log("Conclusion: Groups are NOT balanced.")
            return list(set(imbalanced_features))
        else:
            self.is_balanced = True
            self.log("Conclusion: Groups appear balanced.")
            return []

    def rebalance(self):
        """
        Uses Inverse Probability Weighting (IPW) to balance groups.
        Predicts P(Group | Features) and weights by 1/P.
        """
        if self.is_balanced:
            self.log("Groups are balanced. No re-balancing needed.")
            return

        self.log("Attempting statistical re-balancing using Inverse Probability Weighting (IPW)...")
        
        group_col = self.col_mapping['group']
        features = self.col_mapping['features']
        
        # Prepare data for logistic regression
        # Handle categoricals via one-hot
        X = pd.get_dummies(self.data[features], drop_first=True)
        X = X.fillna(X.mean()) # Simple imputation
        
        y = self.data[group_col]
        
        clf = LogisticRegression(solver='lbfgs', max_iter=1000)
        clf.fit(X, y)
        
        # Get propensity scores
        probs = clf.predict_proba(X)
        classes = clf.classes_
        
        # Assign weights: 1 / P(assigned_class)
        weights = np.zeros(len(y))
        
        for i, label in enumerate(y):
            # Find the column index for this label
            class_idx = np.where(classes == label)[0][0]
            propensity = probs[i, class_idx]
            # Clip propensity to avoid exploding weights
            propensity = max(0.01, min(0.99, propensity))
            weights[i] = 1.0 / propensity
            
        self.data['ipw_weight'] = weights
        self.log("Re-balancing complete. Weights assigned.")
        self.rebalancing_model = clf

    def analyze_stratified(self, alpha=0.05, alternative='two-sided'):
        """
        Analyzes effect size per segment and aggregates them (Stratified Analysis).
        Dynamically chooses DiD or T-Test per segment based on Pre-Experiment Balance.
        Returns DataFrame with detailed and aggregated results.
        """
        group_col = self.col_mapping.get('group')
        metric_col = self.col_mapping.get('metric')
        seg_col = self.col_mapping.get('segment')
        
        # Handle multi-column segment input
        if isinstance(seg_col, list):
            if not seg_col: return None # Empty list
            if len(seg_col) == 1:
                seg_col = seg_col[0]
            else:
                # Combine multiple columns
                combined_name = " + ".join(seg_col)
                self.data[combined_name] = self.data[seg_col].astype(str).agg(' | '.join, axis=1)
                seg_col = combined_name
        pre_metric_col = self.col_mapping.get('pre_metric')
        
        if not all([group_col, metric_col, seg_col]): return None

        groups = self.data[group_col].unique()
        segments = self.data[seg_col].unique()
        
        # Identify Control
        lower_groups = [g.lower() for g in groups]
        if 'control' in lower_groups: control_label = groups[lower_groups.index('control')]
        elif 'ctrl' in lower_groups: control_label = groups[lower_groups.index('ctrl')]
        else: control_label = groups[0]

        results = []
        
        for g in groups:
            if g == control_label: continue
            
            stratified_effect_sum = 0
            stratified_activation_sum = 0 # New aggregation
            total_weight = 0
            combined_var_sum = 0 # To calculate standard error of the weighted average
            
            used_methods = set()
            
            # Per Segment
            for seg in segments:
                seg_df = self.data[self.data[seg_col] == seg]
                
                c_df = seg_df[seg_df[group_col] == control_label]
                t_df = seg_df[seg_df[group_col] == g]
                
                if len(c_df) < 2 or len(t_df) < 2: continue
                
                # 1. Check A/A P-Value (Pre-Experiment Balance)
                aa_p_val = None
                is_imbalanced_pre = False
                
                if pre_metric_col:
                     c_pre = c_df[pre_metric_col]
                     t_pre = t_df[pre_metric_col]
                     if len(c_pre) > 1 and len(t_pre) > 1:
                         try:
                            t_stat_aa, aa_p_val = stats.ttest_ind(c_pre, t_pre, equal_var=False)
                            if not np.isnan(aa_p_val) and aa_p_val < 0.05:
                                is_imbalanced_pre = True
                         except:
                            pass

                # 2. Method Selection per Segment
                if is_imbalanced_pre:
                    # DiD
                    c_vals = c_df[metric_col] - c_df[pre_metric_col]
                    t_vals = t_df[metric_col] - t_df[pre_metric_col]
                    method_name = f"DiD ({alternative})"
                else:
                    # Standard T-Test
                    c_vals = c_df[metric_col]
                    t_vals = t_df[metric_col]
                    method_name = f"T-Test ({alternative})"
                
                used_methods.add(method_name)
                
                # Stats
                mean_c = c_vals.mean()
                mean_t = t_vals.mean()
                effect = mean_t - mean_c
                
                # Variance calculation for aggregation
                var_c = c_vals.var(ddof=1) / len(c_vals) if len(c_vals) > 1 else 0
                var_t = t_vals.var(ddof=1) / len(t_vals) if len(t_vals) > 1 else 0
                var_effect = var_c + var_t
                
                weight = len(seg_df)
                
                stratified_effect_sum += effect * weight
                # Accumulated Variance of sum(w_i * effect_i) = sum(w_i^2 * Var(effect_i))
                combined_var_sum += (weight ** 2) * var_effect
                total_weight += weight
                
                # Significance for this segment
                t_stat, p_val = stats.ttest_ind(t_vals, c_vals, equal_var=False, alternative=alternative)
                
                # --- Activation Effect (Per Segment) ---
                # Check proportion > 0
                count_c = (c_df[metric_col] > 0).sum()
                n_c = len(c_df)
                count_t = (t_df[metric_col] > 0).sum()
                n_t = len(t_df)
                
                activation_effect = 0.0
                prop_pval = 1.0
                
                if n_c > 0 and n_t > 0:
                     # Map scipy 'alternative' to statsmodels 'alternative'
                     prop_alternative = 'two-sided'
                     if alternative == 'greater': prop_alternative = 'larger'
                     elif alternative == 'less': prop_alternative = 'smaller'

                     stat_prop, prop_pval = smprop.test_proportions_2indep(count_t, n_t, count_c, n_c, alternative=prop_alternative, compare='diff', return_results=False)
                     
                     if prop_pval < alpha:
                         prop_diff = (count_t/n_t) - (count_c/n_c)
                         # Use segment control mean
                         activation_effect = mean_c * prop_diff
                
                total_effect = effect + activation_effect
                
                results.append({
                    'treatment': g,
                    'segment_col': seg_col,
                    'segment_value': str(seg),
                    'method': method_name,
                    'control_mean': mean_c,
                    'treatment_mean': mean_t,
                    'avg_diff_effect': effect,
                    'activation_effect': activation_effect,
                    'total_effect': total_effect,
                    'effect_size': effect, # Keep for compat
                    'weight': weight,
                    'p_value': p_val,
                    'prop_p_value': prop_pval,
                    'aa_p_val': aa_p_val,
                    'significant': p_val < alpha
                })
                
                stratified_activation_sum += activation_effect * weight
            
            # Overall Stratified Effect
            if total_weight > 0:
                overall_effect = stratified_effect_sum / total_weight
                overall_activation = stratified_activation_sum / total_weight
                overall_total = overall_effect + overall_activation
                
                # SE of weighted average = sqrt(sum(w_i^2 * var_i)) / sum(w_i)
                overall_se = np.sqrt(combined_var_sum) / total_weight
                
                # Z-Test for aggregated effect
                if overall_se > 0:
                    z_score = overall_effect / overall_se
                    if alternative == 'two-sided':
                        overall_p = 2 * (1 - stats.norm.cdf(abs(z_score)))
                    elif alternative == 'greater':
                        overall_p = 1 - stats.norm.cdf(z_score)
                    else: # less
                        overall_p = stats.norm.cdf(z_score)
                else:
                    overall_p = 1.0

                if len(used_methods) > 1:
                    agg_method = "Mixed (DiD & T-Test)"
                elif used_methods:
                    agg_method = list(used_methods)[0] + " (Stratified)"
                else:
                    agg_method = "Aggregated"

                results.append({
                    'treatment': g,
                    'segment_col': 'OVERALL (Weighted)',
                    'segment_value': 'TOTAL',
                    'method': agg_method,
                    'control_mean': -1, 
                    'treatment_mean': -1,
                    'effect_size': overall_effect,
                    'weight': total_weight,
                    'p_value': overall_p, 
                    'aa_p_val': 1.0, 
                    'significant': overall_p < alpha,
                    'avg_diff_effect': overall_effect,
                    'activation_effect': overall_activation,
                    'total_effect': overall_total,
                    'prop_p_value': 1.0 # Not calculated for overall
                })
                
        return pd.DataFrame(results)

    def analyze_frequentist(self, alpha=0.05, alternative='two-sided'):
        """
        Performs Frequentist analysis. 
        If 'segment' is defined, defaults to the Stratified/Mixed result. 
        Otherwise falls back to Global DiD or Global T-Test.
        """
        group_col = self.col_mapping['group']
        metric_col = self.col_mapping['metric']
        pre_metric_col = self.col_mapping.get('pre_metric')
        seg_col = self.col_mapping.get('segment')
        
        if not group_col or not metric_col: return {}
        
        # 1. Try Stratified Analysis First (Preferred)
        if seg_col:
            # We must be careful to avoid infinite recursion if we called check_balance here, but we are self contained
            strat_df = self.analyze_stratified(alpha=alpha, alternative=alternative)
            
            if strat_df is not None and not strat_df.empty:
                results = {}
                total_rows = strat_df[strat_df['segment_value'] == 'TOTAL']
                
                if not total_rows.empty:
                    self.log(f"Using Stratified Analysis results for main display (Segment: {seg_col}).")
                    
                    # Need control mean for relative effect. 
                    # The stratified total doesn't easily give a 'control mean'.
                    # We will use the global control mean for relative calc.
                    groups = self.data[group_col].unique()
                    lower_groups = [g.lower() for g in groups]
                    if 'control' in lower_groups: control_label = groups[lower_groups.index('control')]
                    elif 'ctrl' in lower_groups: control_label = groups[lower_groups.index('ctrl')]
                    else: control_label = groups[0]
                    
                    control_df_global = self.data[self.data[group_col] == control_label]
                    global_control_mean = control_df_global[metric_col].mean()

                    for _, row in total_rows.iterrows():
                        g = row['treatment']
                        effect_size = row['effect_size']
                        
                        rel_eff = 0
                        if global_control_mean != 0:
                            rel_eff = effect_size / global_control_mean
                            
                        results[g] = {
                            'control': control_label,
                            'treatment': g,
                            'method': row['method'],
                            'avg_diff_effect': row.get('avg_diff_effect', effect_size),
                            'activation_effect': row.get('activation_effect', 0.0),
                            'total_effect': row.get('total_effect', effect_size),
                            'effect_size': effect_size, # Legacy support
                            'relative_effect': rel_eff, 
                            'p_value': row['p_value'],
                            'prop_p_value': row.get('prop_p_value', 1.0), 
                            'significant': row['significant']
                        }
                    return results

        # 2. Fallback: Global Analysis
        # ... (Existing logic) ...
        groups = self.data[group_col].unique()
        results = {}
        
        control_label = groups[0]
        if 'control' in [g.lower() for g in groups]:
             control_label = [g for g in groups if g.lower() == 'control'][0]

        control_df = self.data[self.data[group_col] == control_label]
        
        use_did = getattr(self, 'use_did', False) and pre_metric_col is not None
        c_weights = control_df.get('ipw_weight', np.ones(len(control_df)))

        if use_did:
            c_vals = control_df[metric_col] - control_df[pre_metric_col]
            strategy_name = f"Diff-in-Diff ({alternative})"
        else:
            c_vals = control_df[metric_col]
            strategy_name = f"T-Test ({alternative})"
        
        c_mean = np.average(c_vals, weights=c_weights)
            
        for g in groups:
            if g == control_label: continue
            
            treat_df = self.data[self.data[group_col] == g]
            t_weights = treat_df.get('ipw_weight', np.ones(len(treat_df)))
            
            if use_did:
                t_vals = treat_df[metric_col] - treat_df[pre_metric_col]
            else:
                t_vals = treat_df[metric_col]
            
            t_mean = np.average(t_vals, weights=t_weights)
            
            effect_size = t_mean - c_mean
            
            if control_df[metric_col].mean() != 0:
                 relative_effect = effect_size / control_df[metric_col].mean()
            else:
                 relative_effect = 0
            
            # Significance Test
            # Note: ttest_ind(a, b) tests population mean of a - population mean of b
            # We want Treatment (t_vals) - Control (c_vals). So we pass t_vals first to align with 'alternative' logic (e.g. greater means T > C)
            t_stat, p_val = stats.ttest_ind(t_vals, c_vals, equal_var=False, alternative=alternative)
            
            # --- Activation Effect (Binomial Test) ---
            # Proportion of users with metric > 0
            # Note: We use the raw series for this, assuming 0 means inactive.
            
            # If DiD is used, t_vals/c_vals are differences, so "activation" is tricky.
            # The prompt implies "number of users with higher than 0 value". This usually refers to the POST metric or the raw metric.
            # Assuming we use the metric_col directly for activation check (ignoring DiD for activation definition for now, or using raw).
            # Given "control avg times difference", it usually applies to the level.
            
            # Let's use the actual data slices (without DiD subtraction) to determine "active" status
            c_raw = control_df[metric_col]
            t_raw = treat_df[metric_col]
            
            count_c = (c_raw > 0).sum()
            n_c = len(c_raw)
            count_t = (t_raw > 0).sum()
            n_t = len(t_raw)
            
            activation_effect = 0
            prop_pval = 1.0
            prop_diff = 0
            
            if n_c > 0 and n_t > 0:
                # Map scipy 'alternative' to statsmodels 'alternative'
                # scipy: greater, less, two-sided
                # statsmodels: larger, smaller, two-sided
                prop_alternative = 'two-sided'
                if alternative == 'greater': prop_alternative = 'larger'
                elif alternative == 'less': prop_alternative = 'smaller'

                # statsmodels test_proportions_2indep: count1, nobs1, count2, nobs2
                # We want Treatment vs Control
                stat_prop, prop_pval = smprop.test_proportions_2indep(count_t, n_t, count_c, n_c, alternative=prop_alternative, compare='diff', return_results=False)
                
                prop_t = count_t / n_t
                prop_c = count_c / n_c
                prop_diff = prop_t - prop_c
                
                # Check significance for activation
                if prop_pval < alpha:
                    # Formula: control avg * (prop_t - prop_c)
                    # "control avg" = c_mean (which is weighted if IPW, or raw if not). Using c_mean from above.
                    activation_effect = c_mean * prop_diff
            
            total_effect = effect_size + activation_effect
                
            results[g] = {
                'control': control_label,
                'treatment': g,
                'method': strategy_name,
                'avg_diff_effect': effect_size, # Renamed from effect_size
                'activation_effect': activation_effect,
                'total_effect': total_effect,
                'relative_effect': relative_effect,
                'p_value': p_val,
                'prop_p_value': prop_pval,
                'significant': p_val < alpha
            }
            
        return results

    def _fit_bayesian(self, series, n_samples=10000, seed=42):
        """Helper to fit bayesian model for a series."""
        series = series.dropna()
        is_binary = series.apply(lambda x: float(x).is_integer()).all() and series.nunique() <= 2
        unique_vals = series.unique()
        if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
            is_binary = True

        if is_binary:
            alpha_prior, beta_prior = 1, 1
            conversions = series.sum()
            n = len(series)
            alpha_post = alpha_prior + conversions
            beta_post = beta_prior + (n - conversions)
            return {
                'type': 'beta',
                'alpha': alpha_post,
                'beta': beta_post,
                'mean': alpha_post / (alpha_post + beta_post),
                'samples': stats.beta.rvs(alpha_post, beta_post, size=n_samples, random_state=seed)
            }
        else:
            n = len(series)
            mu = series.mean()
            sigma = series.std()
            # If sigma is 0 (constant value), handle gracefully
            if sigma == 0: sigma = 0.0001
            
            # T-dist for mean
            # posterior mean ~ t(df=n-1, loc=mu, scale=sigma/sqrt(n))
            std_err = sigma / np.sqrt(n)
            return {
                'type': 'normal_t',
                'mean': mu,
                'std_err': std_err,
                'df': n - 1 if n > 1 else 1,
                'samples': stats.t.rvs(n - 1 if n > 1 else 1, loc=mu, scale=std_err, size=n_samples, random_state=seed)
            }

    def analyze_bayesian(self, alternative='two-sided'):
        """
        Bayesian analysis using conjugate priors.
        Returns posteriors samples.
        """
        group_col = self.col_mapping['group']
        metric_col = self.col_mapping['metric']
        seg_col = self.col_mapping.get('segment')
        
        # 1. Prefer Stratified Analysis if segment exists (Unified View)
        if seg_col:
            strat_df, strat_plots = self.analyze_stratified_bayesian(alternative=alternative)
            if strat_df is not None and not strat_df.empty:
                 # Extract TOTAL row for main display
                 total_row = strat_df[strat_df['segment_value'] == 'TOTAL (Weighted)']
                 if not total_row.empty:
                     row = total_row.iloc[0]
                     
                     # Reconstruct results format expected by app.py
                     # We use the plot data from the stratified analysis for the curves
                     # and the dataframe row for the metrics
                     
                     results = strat_plots['TOTAL'] # This contains the distribution params for plots
                     
                     # Construct comparison results
                     comparison_results = {}
                     
                     # We need to find the treatment label. 
                     groups = self.data[group_col].unique()
                     lower_groups = [g.lower() for g in groups]
                     if 'control' in lower_groups: control_label = groups[lower_groups.index('control')]
                     elif 'ctrl' in lower_groups: control_label = groups[lower_groups.index('ctrl')]
                     else: control_label = groups[0]
                     
                     for g in groups:
                         if g == control_label: continue
                         # For now, Stratified Bayesian Logic (above) only supports 1 Treatment vs 1 Control aggregation clearly.
                         # The strat_plots['TOTAL'] has keys for control_label and treatment_label.
                         
                         comparison_results[g] = {
                             'prob_treatment_better': row['prob_treatment_better'],
                             'expected_uplift': row['expected_uplift']
                         }
                     
                     return results, comparison_results

        # 2. Fallback: Global Analysis (Pooled)
        results = {}
        groups = self.data[group_col].unique()
        
        comparison_results = {}
        
        # We need a control reference to compare against
        lower_groups = [g.lower() for g in groups]
        if 'control' in lower_groups: control_label = groups[lower_groups.index('control')]
        elif 'ctrl' in lower_groups: control_label = groups[lower_groups.index('ctrl')]
        else: control_label = groups[0]

        # Fit all groups
        fitted_models = {}
        for g in groups:
            subset = self.data[self.data[group_col] == g][metric_col]
            fitted_models[g] = self._fit_bayesian(subset, seed=42)
            # Remove samples for plotting params, but keep for comparison
            res_summary = fitted_models[g].copy()
            del res_summary['samples']
            results[g] = res_summary
            
        c_samples = fitted_models[control_label]['samples']
        
        for g in groups:
            if g == control_label: continue
            
            t_samples = fitted_models[g]['samples']
            
            if alternative == 'less':
                prob_better = (t_samples < c_samples).mean()
            else:
                prob_better = (t_samples > c_samples).mean()
                
            uplift = (t_samples - c_samples).mean()
            
            comparison_results[g] = {
                'prob_treatment_better': prob_better,
                'expected_uplift': uplift
            }
            
        return results, comparison_results

    def analyze_stratified_bayesian(self, alternative='two-sided'):
        """
        Analyzes Bayesian effect size per segment and aggregates them via simulation.
        Returns:
            - segment_results: DataFrame of per-segment Bayesian stats
            - aggregate_results: Dictionary of aggregated Bayesian stats (Prob > Control, Total Uplift)
            - posterior_plots: Dictionary of plot parameters per segment
        """
        group_col = self.col_mapping.get('group')
        metric_col = self.col_mapping.get('metric')
        seg_col = self.col_mapping.get('segment')

        # Handle multi-column segment input
        if isinstance(seg_col, list):
            if not seg_col: return None, None
            if len(seg_col) == 1:
                seg_col = seg_col[0]
            else:
                combined_name = " + ".join(seg_col)
                self.data[combined_name] = self.data[seg_col].astype(str).agg(' | '.join, axis=1)
                seg_col = combined_name
        
        if not all([group_col, metric_col, seg_col]): return None, None

        groups = self.data[group_col].unique()
        segments = self.data[seg_col].unique()
        
        # Identify Control
        lower_groups = [g.lower() for g in groups]
        if 'control' in lower_groups: control_label = groups[lower_groups.index('control')]
        elif 'ctrl' in lower_groups: control_label = groups[lower_groups.index('ctrl')]
        else: control_label = groups[0]

        segment_results = []
        plot_data = {} # seg -> {group -> params}
        
        # For Aggregation
        n_samples = 10000
        total_c_samples = np.zeros(n_samples)
        total_t_samples = np.zeros(n_samples)
        total_weight = 0
        
        treatment_label = [g for g in groups if g != control_label][0]
        
        for seg in segments:
            seg_df = self.data[self.data[seg_col] == seg]
            weight = len(seg_df)
            total_weight += weight
            
            c_subset = seg_df[seg_df[group_col] == control_label][metric_col].dropna()
            t_subset = seg_df[seg_df[group_col] == treatment_label][metric_col].dropna()
            
            c_count = len(c_subset)
            t_count = len(t_subset)
            
            if c_count < 1 or t_count < 1: continue

            # Fit with deterministic seed
            c_fit = self._fit_bayesian(c_subset, seed=42)
            t_fit = self._fit_bayesian(t_subset, seed=42)
            
            # Per Segment Stats
            if alternative == 'less':
                prob_better = (t_fit['samples'] < c_fit['samples']).mean()
            else:
                prob_better = (t_fit['samples'] > c_fit['samples']).mean()
                
            expected_uplift = (t_fit['samples'] - c_fit['samples']).mean()
            
            segment_results.append({
                'segment_value': str(seg),
                'control_mean': c_fit['mean'],
                'treatment_mean': t_fit['mean'],
                'prob_treatment_better': prob_better,
                'expected_uplift': expected_uplift,
                'control_support': c_count,
                'treatment_support': t_count,
                'total_support': c_count + t_count
            })
            
            # Prepare plotting data (remove samples)
            plot_data[str(seg)] = {
                control_label: {k:v for k,v in c_fit.items() if k!='samples'},
                treatment_label: {k:v for k,v in t_fit.items() if k!='samples'}
            }
            
            # Aggregate samples (Weighted Sum)
            total_c_samples += (c_fit['samples'] * weight)
            total_t_samples += (t_fit['samples'] * weight)
            
        # Final Aggregate Stats
        if total_weight > 0:
            agg_c_samples = total_c_samples / total_weight
            agg_t_samples = total_t_samples / total_weight
            
            agg_c_samples = total_c_samples / total_weight
            agg_t_samples = total_t_samples / total_weight
            
            if alternative == 'less':
                agg_prob_better = (agg_t_samples < agg_c_samples).mean()
            else:
                agg_prob_better = (agg_t_samples > agg_c_samples).mean()
                
            agg_uplift = (agg_t_samples - agg_c_samples).mean()
            
            segment_results.append({
                'segment_value': 'TOTAL (Weighted)',
                'control_mean': agg_c_samples.mean(),
                'treatment_mean': agg_t_samples.mean(),
                'prob_treatment_better': agg_prob_better,
                'expected_uplift': agg_uplift,
                'control_support': '-',
                'treatment_support': '-',
                'total_support': total_weight
            })
            
            # Plot data for TOTAL (Approximated as Normal from CLT)
            plot_data['TOTAL'] = {
                 control_label: {
                     'type': 'normal_t', 
                     'mean': agg_c_samples.mean(), 
                     'std_err': agg_c_samples.std(),
                     'df': 1000
                 },
                 treatment_label: {
                     'type': 'normal_t', 
                     'mean': agg_t_samples.mean(), 
                     'std_err': agg_t_samples.std(),
                     'df': 1000
                 }
            }

        return pd.DataFrame(segment_results), plot_data
