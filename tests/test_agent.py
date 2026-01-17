
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))
print("Importing ABAgent...")
from ab_agent import ABAgent

def test_on_file(filename):
    print(f"\n--- Testing on {filename} ---")
    agent = ABAgent()
    with open(filename, 'r') as f:
        # Load
        status, msg = agent.load_data(f)
        if not status:
            print(f"Error loading: {msg}")
            return
    
    # Heuristics
    mapping = agent.guess_columns()
    
    # Force 'experiment_group' or 'ab_variant' if the heuristic failed for some reason
    # (Though it should work)
    
    # Check Balance
    imbal = agent.check_balance()
    
    # Rebalance if needed
    if not agent.is_balanced:
        agent.rebalance()
        
    # Analyze
    freq = agent.analyze_frequentist()
    print("Frequentist Results (First 2):")
    for k, v in list(freq.items())[:2]:
        print(f"  {k}: Effect={v['effect_size']:.4f}, p={v['p_value']:.4f}")
        
    bayes, comp = agent.analyze_bayesian()
    print("Bayesian Results (First 2):")
    for k, v in list(comp.items())[:2]:
        print(f"  {k} > Control: {v['prob_treatment_better']:.1%} chance")

if __name__ == "__main__":
    test_on_file('sample_ab_data.csv')
    test_on_file('sample_ab_data_alt.csv')
