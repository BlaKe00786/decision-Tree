
from sklearn import datasets



iris = datasets.load_iris()                        # Load iris dataset

X = iris.data[:, :];                           # Assign matrix X
y = iris.target;                                    # Assign vector y


from sklearn.tree import DecisionTreeClassifier    # Import decision tree classifier model
tree = DecisionTreeClassifier(criterion='entropy', # Initialize and fit classifier
    max_depth=40, random_state=1)
tree.fit(X, y)



from pydotplus.graphviz import graph_from_dot_data
from sklearn.tree import export_graphviz

dot_data = export_graphviz(                           # Create dot data
    tree, filled=True, rounded=True,
    class_names=['Setosa', 'Versicolor','Virginica'],
    feature_names=['sepal length', 'sepal width','petal length', 'petal width'],
    out_file=None
)

graph = graph_from_dot_data(dot_data)                 # Create graph from dot data
graph.write_png('tree.png')                           # Write graph to PNG image
