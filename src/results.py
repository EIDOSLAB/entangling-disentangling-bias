import wandb
import pandas as pd
from pprint import pprint
import sys

api = wandb.Api()

def sweep_summary(sweep_id, n_top):
    sweep = api.sweep(sweep_id)
    project = sweep.project
    name = sweep.config['name']

    best_runs = pd.DataFrame()
    for run in sweep.runs:
        history = run.history()
        if 'valid_acc' not in history.columns:
            continue

        best = history.valid_acc.idxmax()
        entry = history.loc[best].copy()
        entry['state'] = run.state
        entry['id'] = run.id
        entry['alpha'] = run.config['alpha']
        entry['beta'] = run.config['beta']
        best_runs = best_runs.append(entry)

    metrics = ['biased_test_acc', 'unbiased_test_acc']
    best_runs = best_runs.sort_values('valid_acc', ascending=False).loc[:, ['valid_acc'] + metrics + ['alpha', 'beta', 'state', 'id']]
    
    print(f'------- SUMMARY FOR {project}/{name} -------')
    print(best_runs)
    
    for metric in metrics:
        print(f'{metric}: {best_runs.iloc[:n_top][metric].mean()*100:.2f} ± {best_runs.iloc[:n_top][metric].std(ddof=0)*100:.2f}')
    print('\n')

if __name__ == '__main__':
    sweep_id = sys.argv[1]
    sweep_summary(sweep_id, n_top=3)
