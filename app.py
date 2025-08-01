import streamlit as st
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

def generate_decision_tree_image(classifier):
    fig, ax = plt.subplots(figsize=(12, 10))
    plot_tree(classifier, filled=True, ax=ax)
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close(fig)
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return img_base64

def main():
    st.title("Decision Tree Classifier")

    st.markdown("""
    Welcome to the Decision Tree Classifier! Here, you can generate and visualize datasets, 
    adjust hyperparameters, and run the decision tree model. Start by selecting the number of samples 
    and clusters to create a dataset. The default dataset consists of 800 samples and 1 cluster per class.
    Tune the hyperparameters to see how they affect the model's performance.
    """)

    # Sidebar inputs for dataset generation
    st.sidebar.header("Generate Dataset")
    n_samples = st.sidebar.number_input("Number of Samples", min_value=1, value=800, step=1)
    n_clusters_per_class = st.sidebar.number_input("Number of Clusters per Class", min_value=1, max_value=2, value=1, step=1)

    # Generate dataset
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

    st.subheader("Scatter Plot of Dataset")
    st.plotly_chart(scatter_fig, use_container_width=True)

    st.sidebar.header("Select Hyperparameters")
    criterion = st.sidebar.selectbox("Criterion", options=["gini", "entropy", "log_loss"], index=0)
    splitter = st.sidebar.selectbox("Splitter", options=["best", "random"], index=0)
    max_depth_input = st.sidebar.number_input("Max Depth (leave as 0 for None)", min_value=0, value=0, step=1)
    max_depth = max_depth_input if max_depth_input > 0 else None
    min_samples_split = st.sidebar.number_input("Min Samples Split", min_value=2, value=2, step=1)
    min_samples_leaf = st.sidebar.number_input("Min Samples Leaf", min_value=1, value=1, step=1)
    max_features = st.sidebar.selectbox("Max Features", options=[1, 2], index=1)
    max_leaf_nodes = st.sidebar.number_input("Max Leaf Nodes", min_value=0, value=0, step=1)
    max_leaf_nodes = max_leaf_nodes if max_leaf_nodes > 0 else None
    min_impurity_decrease = st.sidebar.number_input("Min Impurity Decrease", min_value=0.0, value=0.0, step=0.01, format="%.2f")

    if st.sidebar.button("Run Model"):
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
        report = classification_report(y_test, y_pred)

        st.subheader("Accuracy Report")
        st.text(f"Accuracy: {accuracy:.2f}")
        st.text(report)

        # Plot decision boundary
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
        Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        contour = go.Contour(
            z=Z,
            x=np.linspace(x_min, x_max, Z.shape[1]),
            y=np.linspace(y_min, y_max, Z.shape[0]),
            colorscale='cividis',
            opacity=0.3,
            showscale=False,
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

        st.subheader("Decision Tree Classification Plot")
        st.plotly_chart(classified_region_fig, use_container_width=True)

        # Generate decision tree visualization
        img_base64 = generate_decision_tree_image(classifier)
        st.subheader("Decision Tree Visualization")
        st.image(f"data:image/png;base64,{img_base64}", use_container_width=True)

if __name__ == "__main__":
    main()
