from pycaret.clustering import setup, create_model, pull
import pandas as pd

def run_clustering(data, preprocess_args=None, num_clusters=None):
    """
    Run clustering with optional preprocessing techniques and specified number of clusters.
    
    Args:
    - data: DataFrame, the dataset to be used for clustering.
    - preprocess_args: dict, preprocessing arguments such as normalize, transformation, etc.
    - num_clusters: list, numbers of clusters to try.
    
    Returns:
    - DataFrame, results of clustering evaluation.
    """
    if preprocess_args:
        setup(data=data, **preprocess_args)
    else:
        setup(data=data)
    
    results = pd.DataFrame()
    for n in num_clusters:
        model = create_model('kmeans', num_clusters=n)
        evaluation_results = pull()
        evaluation_results_df = pd.DataFrame(evaluation_results)
        evaluation_results_df['num_clusters'] = n
        results = pd.concat([evaluation_results_df, results], axis=0, ignore_index=True)
    
    results.sort_values(by='num_clusters', ascending=True, inplace=True)
    
    return results


def create_plots(model, plot_types):
    """
    Create plots for clustering evaluation.
    
    Args:
    - model: model, the clustering model.
    - plot_types: list, types of plots to create.
    """
    for pt in plot_types:
        create_model(model, plot=pt)
    



def process_results(results, num_clusters, method_name):
    """
    Process clustering results and organize into DataFrames.
    
    Args:
    - results: DataFrame, clustering evaluation results.
    - num_clusters: list, numbers of clusters used.
    - method_name: str, name of the preprocessing method.
    
    Returns:
    - DataFrame, organized results.
    """
    processed_results = {}
    for n in num_clusters:
        processed_results[f'c{n}'] = results[results['num_clusters'] == n].melt(id_vars='num_clusters', var_name='metric', value_name='value').drop('num_clusters', axis=1)
    
    return pd.DataFrame({f'{method_name}_{k}': pd.Series(v) for k, v in processed_results.items()}, index=[method_name])

