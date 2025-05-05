#include <iostream>
#include <vector>
#include <stack>
#include <omp.h> // OpenMP header
using namespace std;

class Graph {
public:
    // Print adjacency list
    void printAdjList(const vector<vector<int>> &adj) {
        cout << "\nAdjacency List:\n";
        for (int i = 0; i < adj.size(); i++) {
            cout << i << ": ";
            for (int j : adj[i]) {
                cout << j << " ";
            }
            cout << endl;
        }
    }

    // Iterative DFS using stack and OpenMP
    void dfs(vector<vector<int>> &adj, int start) {
        int V = adj.size();
        vector<bool> visited(V, false);
        stack<int> st;

        st.push(start);
        visited[start] = true;

        cout << "\nParallel Iterative DFS Traversal from node " << start << ": ";

        while (!st.empty()) 
        {
            int node;

            #pragma omp critical
            {
                node = st.top();
                st.pop();
                cout << node << " ";
            }

            // Traverse neighbors in parallel
            #pragma omp parallel for
            for (int i = 0; i < adj[node].size(); i++) {
                int neighbor = adj[node][i];

                #pragma omp critical
                {
                    if (!visited[neighbor]) {
                        visited[neighbor] = true;
                        st.push(neighbor);
                    }
                }
            } 
        }
        cout << endl;
    }
};

int main() {
    int V, E;
    cout << "Enter number of vertices: ";
    cin >> V;
    cout << "Enter number of edges: ";
    cin >> E;

    vector<vector<int>> adj(V);
    cout << "Enter edges (u v):\n";
    for (int i = 0; i < E; i++) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u); // for undirected graph
    }

    int start;
    cout << "Enter starting node for DFS: ";
    cin >> start;

    Graph g;
    g.printAdjList(adj);
    g.dfs(adj, start);

    return 0;
}
