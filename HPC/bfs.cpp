#include <iostream>
#include <vector>
#include <queue>
#include <omp.h> 
using namespace std;

class Graph {
public:
    // Function to print the adjacency list
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

    // Parallel BFS using OpenMP
    void bfs(vector<vector<int>> &adj, int start) {
        queue<int> q;

        vector<bool> visited(adj.size(), false);
        
        q.push(start);
        visited[start] = true;

        cout << "\nParallel BFS Traversal starting from node " << start << ": ";

        while (!q.empty()) {
            
            int qsize = q.size(); // Get current level size
            vector<int> currentLevel;

            // Collect all nodes at current level
            for (int i = 0; i < qsize; i++) {
                int node = q.front();
                q.pop();
                cout << node << " ";
                currentLevel.push_back(node);
            }

            // Parallelize neighbor visiting using OpenMP
            #pragma omp parallel for shared(visited, adj, q)
            for (int i = 0; i < currentLevel.size(); i++) 
            {
                int node = currentLevel[i];

                for (int j = 0; j < adj[node].size(); j++) 
                {
                    int neighbor = adj[node][j];
                    // Use atomic to avoid race condition on visited[]
                    #pragma omp critical
                    {
                        if (!visited[neighbor]) {
                            visited[neighbor] = true;
                            q.push(neighbor);
                        }
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
        adj[v].push_back(u); // Remove this for directed graph
    }

    int start;
    cout << "Enter starting node for BFS: ";
    cin >> start;

    Graph g;
    g.printAdjList(adj);
    g.bfs(adj, start);

    return 0;
}
