#include <omp.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <iomanip>

#define nprocs 128

using namespace std;

class Similarity
{
private:
    double decay_factor;
    vector<string> name_list;                  // node name
    unordered_map<string, int> name_index_map; // node name : index
    vector<vector<double>> old_sim;
    vector<vector<double>> new_sim;

public:
    Similarity(const vector<pair<string, string>> &edge_list, double decay_factor) : decay_factor(decay_factor)
    {
        init_sim(edge_list);
        int node_num = name_list.size();                                         // number of nodes
        new_sim = vector<vector<double>>(node_num, vector<double>(node_num, 0)); // initialize
    }

    void init_sim(const vector<pair<string, string>> &edge_list) // to set name_list and name_index_map
    {
        for (const auto &edge : edge_list)
        {
            if (name_index_map.find(edge.first) == name_index_map.end())
            {
                name_index_map[edge.first] = name_list.size();
                name_list.push_back(edge.first);
            }
            if (name_index_map.find(edge.second) == name_index_map.end())
            {
                name_index_map[edge.second] = name_list.size();
                name_list.push_back(edge.second);
            }
        }
        int node_num = name_list.size(); // number of nodes
        old_sim = vector<vector<double>>(node_num, vector<double>(node_num, 0));

        for (int i = 0; i < node_num; ++i)
        {
            old_sim[i][i] = 1.0;
        }

        // Initial similarity matrix set, where diagonals = 1, rest = 0
    }

    void SimRank_one_iter(const vector<pair<string, string>> &edge_list) // One iteration using formula
    {
#pragma omp parallel for collapse(2)
        for (int i = 0; i < name_list.size(); ++i)
        {
            for (int j = 0; j < name_list.size(); ++j)
            {
                new_sim[i][j] = 0.0;
            }
        }

#pragma omp parallel for collapse(2)
        for (int i = 0; i < name_list.size(); ++i)
        {
            for (int j = 0; j < name_list.size(); ++j)
            {
                double new_SimRank = calculate_SimRank(edge_list, name_list[i], name_list[j]);
#pragma omp atomic
                new_sim[i][j] += new_SimRank;
            }
        }
    }

    // calculate using formula
    double calculate_SimRank(const vector<pair<string, string>> &edge_list, const string &node1_name, const string &node2_name)
    {
        if (node1_name == node2_name)
        {
            return 1.0;
        }

        int in_neighbors1 = 0, in_neighbors2 = 0;

        for (const auto &edge : edge_list)
        {
            if (edge.second == node1_name)
            {
                ++in_neighbors1;
            }
            if (edge.second == node2_name)
            {
                ++in_neighbors2;
            }
        }

        if (in_neighbors1 == 0 || in_neighbors2 == 0)
        {
            return 0.0;
        }

        double SimRank_sum = 0;
        for (const auto &edge1 : edge_list)
        {
            if (edge1.second == node1_name)
            {
                for (const auto &edge2 : edge_list)
                {
                    if (edge2.second == node2_name)
                    {
                        SimRank_sum += old_sim[name_index_map[edge1.first]][name_index_map[edge2.first]];
                    }
                }
            }
        }

        double scale = decay_factor / (in_neighbors1 * in_neighbors2);
        double new_SimRank = scale * SimRank_sum;

        return new_SimRank;
    }

    vector<vector<double>> getNewSimilarityMatrix()
    {
        return new_sim;
    }

    void updateOldSimilarityMatrix()
    {
        old_sim = new_sim;
    }
};

vector<pair<string, string>> readEdgeListFromFile(const string &filename) // save in file
{
    vector<pair<string, string>> edge_list;
    ifstream file(filename);
    string line;
    while (getline(file, line))
    {
        istringstream iss(line);
        string node1, node2;
        if (!(iss >> node1 >> node2))
        {
            break;
        }
        edge_list.emplace_back(node1, node2);
    }
    return edge_list;
}

int main()
{
    omp_set_num_threads(nprocs);

    string filename = "chonyy_g_4.txt"; // set filename here

    vector<pair<string, string>> edge_list = readEdgeListFromFile(filename);

    vector<vector<double>> final_similarity_matrix;

    int num_iterations = 100;        // set num_iterations here
    const double DECAY_FACTOR = 0.9; // set decay factor here

    Similarity similarity(edge_list, DECAY_FACTOR);

    auto start_total = chrono::steady_clock::now();

    for (int iter = 0; iter < num_iterations; ++iter) // runs for n iterations
    {
        auto start = chrono::steady_clock::now();

        similarity.SimRank_one_iter(edge_list);

        auto end = chrono::steady_clock::now();
        chrono::duration<double> diff = end - start;

        vector<vector<double>> similarityMatrix = similarity.getNewSimilarityMatrix();

        if (iter == num_iterations - 1)
        {
            final_similarity_matrix = similarityMatrix;
        }

        similarity.updateOldSimilarityMatrix();
    }

    auto end_total = chrono::steady_clock::now();
    chrono::duration<double> diff_total = end_total - start_total;

    cout << "Total time to complete all iterations: " << diff_total.count() << " seconds" << endl; // print total time

    ostringstream oss; // save file name as Sim_result_{filename}-{num_iterations}
    oss << "Sim_result_chonyy_g_4.txt-" << num_iterations << "_iter";
    ofstream outfile(oss.str());

    if (outfile.is_open())
    {
        for (const auto &row : final_similarity_matrix)
        {
            for (const auto &val : row)
            {
                outfile << val << "\t";
            }
            outfile << "\n";
        }
        outfile.close();
    }
    else
    {
        cerr << "Unable to open file for writing!" << endl;
    }

    return 0;
}
