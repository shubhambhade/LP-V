#include<iostream>
#include<queue>
#include<omp.h>
using namespace std;

class Node
{
public:
    int value;
    Node *left;
    Node *right;

    Node(int value)
    {
        this->value = value;
        this->left = NULL;
        this->right = NULL;
    }
};

Node *generateTree(int values[], int length)
{
    if (length == 0)
    {
        return NULL;
    }

    Node* treeNode[length];

    for (int i = 0; i < length; i++)
    {
        if (values[i] != -1)
        {
            treeNode[i] = new Node(values[i]);
        }
        else
        {
            treeNode[i] = NULL;
        }
    }

    int parent = 0;
    int child = 1;

    while (child < length)
    {
        if (treeNode[parent] != NULL)
        {
            treeNode[parent]->left = treeNode[child++];
            if (child < length)
            {
                treeNode[parent]->right = treeNode[child++];
            }
        }
        parent++;
    }

    Node *root = treeNode[0];
    return root;
}

void bfs(Node *root)
{
    if (root == NULL)
    {
        return;
    }

    queue<Node *> q;
    q.push(root);

    cout << "BFS Traversal: ";
    while (!q.empty())
    {
        Node *node = q.front();
        q.pop();
        cout << node->value << " => ";

        if (node->left != NULL)
        {
            q.push(node->left);
        }
        if (node->right != NULL)
        {
            q.push(node->right);
        }
    }
    cout << "NULL" << endl;
}

void dfs(Node *root)
{
    if (root == NULL)
    {
        return;
    }

    #pragma omp critical
    cout << root->value << " => ";

    #pragma omp parallel sections
    {
        #pragma omp section
        dfs(root->left);

        #pragma omp section
        dfs(root->right);
    }
}

int main()
{
    int length = 16;
    int values[length];
    for (int i = 0; i < length; i++)
    {
        values[i] = i;
    }

    Node *root = generateTree(values, length);

    bfs(root);

    cout << "DFS Traversal: ";
    dfs(root);
    cout << "NULL" << endl;

    return 0;
}

