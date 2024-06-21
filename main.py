from flask import Flask, render_template, request, session
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import base64
from io import BytesIO

app = Flask(__name__)

# Default synthetic data parameters
n_samples_default = 800
n_clusters_per_class_default = 1

# Initial data and scatter plot
X, y = make_classification(n_samples=n_samples_default, n_features=2, n_informative=2,
                           n_redundant=0, n_clusters_per_class=n_clusters_per_class_default, random_state=9)
df = pd.DataFrame(X, columns=['Feature 1', 'Feature 2'])
df['target'] = y

scatter_fig = px.scatter(df, x='Feature 1', y='Feature 2', color='target')
scatter_fig.update_layout(
    xaxis_title='Feature 1',
    yaxis_title='Feature 2',
    margin=dict(l=0, r=0, t=30, b=0),
    width=800,
    height=600,
)


# Function to generate the decision tree image
def generate_decision_tree_image(classifier):
    fig, ax = plt.subplots(figsize=(12, 10))
    plot_tree(classifier, filled=True, ax=ax)
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close(fig)
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return 'data:image/png;base64,{}'.format(img_base64)


#home route
@app.route('/')
def index():                                    
    return render_template('index.html', scatter_plot=scatter_fig.to_html(include_plotlyjs='cdn'))


#route to generate dataset
@app.route('/generate_dataset', methods=['POST'])
def generate_dataset():
    global X, y, df, scatter_fig, n_samples, n_samples_default
    
    n_samples_str = request.form['n_samples']
    if n_samples_str:
        n_samples = int(n_samples_str)
    else:
        n_samples = 800 
   
    n_clusters_per_class = request.form.get('n_clusters_per_class')
    if n_clusters_per_class:
        n_clusters_per_class = int(n_clusters_per_class)
    else:
        n_clusters_per_class = 1
    
    try:
        X, y = make_classification(n_samples=n_samples, n_features=2, n_informative=2,
                                   n_redundant=0, n_clusters_per_class=n_clusters_per_class, random_state=9)
        df = pd.DataFrame(X, columns=['Feature 1', 'Feature 2'])
        df['target'] = y
        
        scatter_fig = px.scatter(df, x='Feature 1', y='Feature 2', color='target')
        scatter_fig.update_layout(
            xaxis_title='Feature 1',
            yaxis_title='Feature 2',
            margin=dict(l=0, r=0, t=30, b=0),
            width=800,
            height=600,
        )
    
        n_samples_default = n_samples
        n_clusters_per_class_default = n_clusters_per_class

    except ValueError as e:
        error_message = "Number of clusters per class cannot exceed 2."
        return render_template('index.html', error_message=error_message)
    
    return render_template('index.html', scatter_plot=scatter_fig.to_html(include_plotlyjs='cdn'),
                           n_samples=n_samples_default,
                           n_clusters_per_class=n_clusters_per_class_default)



#route to run model with selected hyperparameters
@app.route('/run_model', methods=['POST'])
def run_model():
    global X, y, df
    
    # criterion, splitter
    criterion = request.form.get('criterion', 'gini')
    splitter = request.form.get('splitter', 'best')
    
    # max_depth
    max_depth = request.form.get('max_depth')
    if max_depth:
        max_depth = int(max_depth)
    else:
        max_depth = None
    
    # min_samples_split, min_samples_leaf, max_features
    min_samples_split = int(request.form.get('min_samples_split', '2'))
    min_samples_leaf = int(request.form.get('min_samples_leaf', '1'))
    max_features = int(request.form.get('max_features', '2'))
    
    # max_leaf_nodes
    max_leaf_nodes = request.form.get('max_leaf_nodes')
    if max_leaf_nodes:
        max_leaf_nodes = int(max_leaf_nodes)
    else:
        max_leaf_nodes = None
    
    # min_impurity_decrease
    min_impurity_decrease = request.form.get('min_impurity_decrease')
    if min_impurity_decrease:
        min_impurity_decrease = float(min_impurity_decrease)
    else:
        min_impurity_decrease = 0.0  # Default value
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)
    
    # Initialize and fit the decision tree classifier
    classifier = DecisionTreeClassifier(criterion=criterion,
                                        splitter=splitter,
                                        max_depth=max_depth,
                                        min_samples_split=min_samples_split,
                                        min_samples_leaf=min_samples_leaf,
                                        max_features=max_features,
                                        max_leaf_nodes=max_leaf_nodes,
                                        min_impurity_decrease=min_impurity_decrease)
    classifier.fit(X_train, y_train)
    
    # Evaluate classifier on test set
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = f'Accuracy: {accuracy:.2f}\n\n{classification_report(y_test, y_pred)}'
    
    # Plot decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    # Create classified plot
    contour = go.Contour(
            z=Z,
            x=np.linspace(x_min, x_max, Z.shape[1]),  # Use linspace for even spacing
            y=np.linspace(y_min, y_max, Z.shape[0]),
            colorscale='cividis',
            opacity=0.3,
            showscale=False,  # hide the color scale for cleaner visualization
            hoverinfo='skip'
        )

    scatter = px.scatter(df, x='Feature 1', y='Feature 2', color='target',
                            color_continuous_scale='cividis', opacity=0.8)

    classified_region_fig = go.Figure(data=[contour, scatter.data[0]])
    classified_region_fig.update_layout(
        title='Decision Tree Classification Boundaries with Data Points',
        xaxis_title='Feature 1',
        yaxis_title='Feature 2',
        margin=dict(l=0, r=0, t=30, b=0),
        width=800,
        height=600,
    )
    classified_region_fig.update_coloraxes(showscale=False)
    
    # Generate decision tree visualization
    decision_tree_image = generate_decision_tree_image(classifier)
    
    return render_template('index.html', 
                           scatter_plot=scatter_fig.to_html(include_plotlyjs='cdn'),
                           classified_plot=classified_region_fig.to_html(include_plotlyjs='cdn'), 
                           report=report, 
                           decision_tree_image=decision_tree_image,
                           criterion=criterion, splitter=splitter, max_depth=max_depth, min_samples_split=min_samples_split,
                           min_samples_leaf=min_samples_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes,
                           min_impurity_decrease=min_impurity_decrease,
                           n_samples=session.get('n_samples', n_samples_default),
                           n_clusters_per_class=session.get('n_clusters_per_class', n_clusters_per_class_default))


# run the app
if __name__ == '__main__':
    app.run(debug=True, threaded=False)
