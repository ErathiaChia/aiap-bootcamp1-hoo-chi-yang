# pipeline.py

from dataloader import DataLoader
from preprocessing import Preprocessor
from build_models import ModelBuilder
from visualize import Visualizer
from model_mgr import Model_Mgr


def main():
    # Load data
    manager = Model_Mgr()
    seed = manager.get_seed()
    print(f'Seed: {seed}')
    
    print('===============================')
    print('Loading data...')
    loader = DataLoader('data/score.db')
    df = loader.import_data()

    print('Dataload completed!')

    print('===============================')
    print('Preprocessing data...')
    # Preprocess data
    preprocessor = Preprocessor(df, seed = seed)
    X_train, X_test, y_train, y_test, __ = preprocessor.preprocess_dataframe()
    print('Preprocessing completed!')
    
    print('===============================')
    print('Building models...')
    # Train and evaluate models
    model_builder = ModelBuilder(X_train, X_test, y_train, y_test, seed = seed)
    best_models, test_scores = model_builder.train_and_evaluate()
    print('Model building completed!')

    print('===============================')
    print('Output Results...')
    # Visualize model performance
    Visualizer.plot_model_performance(test_scores)
    print('Results output completed!')

if __name__ == "__main__":
    main()
